"""Room-native Aurora worker: a real LiveKit participant, not an HTTP bridge.

`talk_server.py` keeps the audio path in the browser and uses the room for
identity only, which the README calls out as the remaining gap. This worker
inverts that:

    caller mic -> LiveKit room -> worker subscribes -> VAD -> STT -> Agent
              -> TTS -> worker publishes track -> LiveKit room -> caller

The room is now the transport rather than decoration, which is what lets a SIP
caller later join as an ordinary participant instead of a special case.

    python agent_worker.py --probe          # prove inbound audio only
    python agent_worker.py                  # run the live loop

Requires a LiveKit server (`./start_local_server.sh`) and a participant
publishing microphone audio (the browser client, or `sim_caller.py`).
"""

from __future__ import annotations

import argparse
import array
import asyncio
import contextlib
import io
import math
import os
import sys
import time
import warnings
import wave
from collections import deque
from pathlib import Path

import jwt
from livekit import api, rtc

from env_loader import load_env_files

ROOT = Path(__file__).resolve().parent
ASSIGNMENT_ROOT = ROOT.parent
PIPELINE_ROOT = ASSIGNMENT_ROOT / "pipeline"

# The cascade runs on 16 kHz mono. webrtcvad accepts 10/20/30 ms frames, so
# asking the SDK for 20 ms gives us 320 samples (640 bytes) per frame -- exactly
# one VAD window, with no resampling or re-buffering on our side.
SAMPLE_RATE = 16000
NUM_CHANNELS = 1
FRAME_MS = 20
FRAME_BYTES = SAMPLE_RATE * 2 * FRAME_MS // 1000

# Speech must persist this long before a turn opens, so a cough or a packet of
# line noise does not start a turn.
SPEECH_ONSET_FRAMES = 3
# Frames of audio kept before onset is confirmed, so the first syllable of the
# utterance is not clipped off the front.
PREROLL_FRAMES = 10
# Utterances shorter than this are treated as noise rather than speech.
MIN_UTTERANCE_MS = 300

AGENT_IDENTITY = os.getenv("AGENT_IDENTITY", "aurora")
AGENT_NAME = os.getenv("AGENT_NAME", "Aurora")
GREETING = "Thanks for calling Aurora Hotel reservations. How can I help?"


# --- connection helpers (kept in step with talk_server.py) ---

def _livekit_url() -> str:
    """Normalize the configured URL to the websocket scheme the SDK expects."""
    raw = os.getenv("LIVEKIT_URL", "ws://localhost:7880")
    if raw.startswith("http://"):
        return "ws://" + raw[len("http://"):]
    if raw.startswith("https://"):
        return "wss://" + raw[len("https://"):]
    return raw


def _token(identity: str, name: str, room: str) -> str:
    """Mint a join token. Mirrors talk_server._token so both agree on grants."""
    secret = os.getenv("LIVEKIT_API_SECRET", "secret")
    if secret == "secret":
        warnings.filterwarnings("ignore", category=jwt.InsecureKeyLengthWarning)
    return (
        api.AccessToken(os.getenv("LIVEKIT_API_KEY", "devkey"), secret)
        .with_identity(identity)
        .with_name(name)
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=room,
                can_publish=True,
                can_subscribe=True,
            )
        )
        .to_jwt()
    )


# --- audio helpers ---

def _rms(pcm: bytes) -> float:
    """Rough loudness of a 16-bit mono frame, 0.0-1.0."""
    samples = array.array("h")
    samples.frombytes(pcm)
    if not samples:
        return 0.0
    mean_square = sum(s * s for s in samples) / len(samples)
    return math.sqrt(mean_square) / 32768.0


def wav_to_pcm(wav_bytes: bytes, target_rate: int = SAMPLE_RATE) -> bytes:
    """Decode WAV to 16-bit mono PCM at `target_rate`.

    Provider TTS returns whatever rate the vendor picked, while the published
    track has one fixed rate, so anything that does not match is resampled.
    """
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        channels, width, rate = w.getnchannels(), w.getsampwidth(), w.getframerate()
        frames = w.readframes(w.getnframes())

    if width != 2:
        raise ValueError(f"Expected 16-bit PCM WAV, got {width * 8}-bit")

    samples = array.array("h")
    samples.frombytes(frames)

    if channels > 1:  # average to mono
        samples = array.array(
            "h",
            [
                sum(samples[i + c] for c in range(channels)) // channels
                for i in range(0, len(samples) - channels + 1, channels)
            ],
        )

    if rate != target_rate:
        import numpy as np

        source = np.frombuffer(samples.tobytes(), dtype=np.int16)
        count = int(len(source) * target_rate / rate)
        resampled = np.interp(
            np.linspace(0, len(source) - 1, count),
            np.arange(len(source)),
            source.astype(np.float32),
        )
        return resampled.astype(np.int16).tobytes()

    return samples.tobytes()


class Endpointer:
    """Turn a stream of 20 ms frames into completed caller utterances.

    Mirrors the endpointing policy in voice_loop.py so the room-native path and
    the local microphone path commit turns on the same rules.
    """

    def __init__(self, aggressiveness: int, silence_ms: int) -> None:
        import webrtcvad

        self.vad = webrtcvad.Vad(aggressiveness)
        self.silence_ms = silence_ms
        self.preroll: deque[bytes] = deque(maxlen=PREROLL_FRAMES)
        self.buffer: list[bytes] = []
        self.capturing = False
        self._onset = 0
        self._trailing_silence = 0
        self._speech_frames = 0

    def push(self, frame: bytes) -> bytes | None:
        """Feed one frame. Returns the utterance PCM when a turn completes."""
        speech = self.vad.is_speech(frame, SAMPLE_RATE)

        if not self.capturing:
            self.preroll.append(frame)
            self._onset = self._onset + 1 if speech else 0
            if self._onset >= SPEECH_ONSET_FRAMES:
                # Include the pre-roll so the leading syllable survives.
                self.capturing = True
                self.buffer = list(self.preroll)
                self.preroll.clear()
                self._speech_frames = self._onset
                self._onset = 0
                self._trailing_silence = 0
            return None

        self.buffer.append(frame)
        if speech:
            self._speech_frames += 1
        self._trailing_silence = 0 if speech else self._trailing_silence + FRAME_MS
        if self._trailing_silence < self.silence_ms:
            return None

        pcm = b"".join(self.buffer)
        # Measure the speech itself, not the buffer. The buffer also holds the
        # pre-roll and the full endpoint silence, so a short cough would
        # otherwise clear the threshold and bill a transcription request.
        speech_ms = self._speech_frames * FRAME_MS
        self.reset()
        if speech_ms < MIN_UTTERANCE_MS:
            return None
        return pcm

    def speech_started(self, frame: bytes) -> bool:
        """True when a frame looks like the caller starting to talk.

        Used for barge-in, where we must react while the agent is still
        speaking rather than waiting for a full endpoint.
        """
        return self.vad.is_speech(frame, SAMPLE_RATE)

    def reset(self) -> None:
        self.capturing = False
        self.buffer = []
        self.preroll.clear()
        self._onset = 0
        self._trailing_silence = 0
        self._speech_frames = 0


class AuroraWorker:
    """Join a room, listen on a subscribed track, answer on a published track."""

    def __init__(self, room_name: str, provider_name: str, greet: bool = True) -> None:
        self.room_name = room_name
        self.provider_name = provider_name
        self.greet = greet
        self.room: rtc.Room | None = None
        self.source: rtc.AudioSource | None = None
        self.agent = None
        self._stream_task: asyncio.Task | None = None
        self._speak_task: asyncio.Task | None = None
        self._turn_task: asyncio.Task | None = None
        self._closing: asyncio.Event | None = None
        self.session_id = f"room-{room_name}"
        self.turns = 0
        self.barge_ins = 0

    # --- lifecycle ---

    async def run(self) -> int:
        # rtc.Room binds asyncio.get_event_loop() in its constructor, which
        # raises on Python 3.14 outside a running loop, so build it here.
        self.room = rtc.Room()
        self._closing = asyncio.Event()
        self.agent = await asyncio.to_thread(self._build_agent)
        self._wire_events()

        url = _livekit_url()
        print(f"Connecting to {url} as {AGENT_IDENTITY!r} in room {self.room_name!r}")
        await self.room.connect(url, _token(AGENT_IDENTITY, AGENT_NAME, self.room_name))
        print(f"Connected. Provider: {self.provider_name}")

        # Publish before listening so the caller has somewhere to hear a reply
        # even if they start talking immediately.
        self.source = rtc.AudioSource(SAMPLE_RATE, NUM_CHANNELS)
        track = rtc.LocalAudioTrack.create_audio_track("aurora-voice", self.source)
        await self.room.local_participant.publish_track(
            track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        )
        print("Published agent audio track 'aurora-voice'")

        # A participant who published before we joined fires no event.
        for participant in self.room.remote_participants.values():
            for publication in participant.track_publications.values():
                if publication.track is not None:
                    self._adopt(publication.track, participant)

        if self.greet:
            await self._speak(GREETING, turn_id="greeting")

        print("Listening. Ctrl-C to stop.")
        await self._closing.wait()
        await self._shutdown()
        print(f"\nTurns handled: {self.turns}  Barge-ins: {self.barge_ins}")
        return 0

    def _build_agent(self):
        if str(PIPELINE_ROOT) not in sys.path:
            sys.path.insert(0, str(PIPELINE_ROOT))
        from agent import Agent
        from providers import make_provider

        return Agent(make_provider(self.provider_name))

    def _trace(self, turn_id: str | None = None):
        if str(PIPELINE_ROOT) not in sys.path:
            sys.path.insert(0, str(PIPELINE_ROOT))
        from telemetry import TurnTrace

        return TurnTrace(session_id=self.session_id, turn_id=turn_id)

    def _wire_events(self) -> None:
        @self.room.on("participant_connected")
        def _on_join(participant: rtc.RemoteParticipant) -> None:
            print(f"  [room] participant joined: {participant.identity}")

        @self.room.on("participant_disconnected")
        def _on_leave(participant: rtc.RemoteParticipant) -> None:
            print(f"  [room] participant left: {participant.identity}")

        @self.room.on("track_subscribed")
        def _on_track(
            track: rtc.Track,
            publication: rtc.RemoteTrackPublication,
            participant: rtc.RemoteParticipant,
        ) -> None:
            self._adopt(track, participant)

    def _adopt(self, track: rtc.Track, participant: rtc.RemoteParticipant) -> None:
        if track.kind != rtc.TrackKind.KIND_AUDIO:
            return
        if participant.identity == AGENT_IDENTITY:
            return  # never listen to ourselves
        if self._stream_task is not None and not self._stream_task.done():
            print(f"  [room] ignoring extra audio track from {participant.identity}")
            return
        print(f"  [room] subscribed to audio from {participant.identity!r}")
        self._stream_task = asyncio.create_task(self._consume(track))

    # --- inbound audio ---

    async def _consume(self, track: rtc.Track) -> None:
        """Frame loop: VAD in, completed utterances out.

        This coroutine must never block. STT, the LLM, and TTS all run in
        worker threads or separate tasks so frames keep flowing, which is what
        makes barge-in possible mid-response.
        """
        endpointer = Endpointer(
            int(os.getenv("VAD_AGGRESSIVENESS", "2")),
            int(os.getenv("ENDPOINT_SILENCE_MS", "600")),
        )
        stream = rtc.AudioStream(
            track,
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
            frame_size_ms=FRAME_MS,
        )
        try:
            async for event in stream:
                frame = bytes(event.frame.data)
                if len(frame) != FRAME_BYTES:
                    continue  # ignore any odd-sized frame the SDK emits

                # Barge-in: the caller talking over the agent cancels playback
                # immediately, before the utterance is even complete.
                if self._is_speaking() and endpointer.speech_started(frame):
                    await self._barge_in()

                utterance = endpointer.push(frame)
                if utterance is None:
                    continue
                if self._turn_task is not None and not self._turn_task.done():
                    continue  # a turn is already in flight
                self._turn_task = asyncio.create_task(self._handle_turn(utterance))
        except asyncio.CancelledError:
            pass
        finally:
            await stream.aclose()

    # --- turn handling ---

    async def _handle_turn(self, pcm: bytes) -> None:
        duration = len(pcm) / FRAME_BYTES * FRAME_MS / 1000
        self.turns += 1
        turn_id = f"turn-{self.turns}"
        trace = self._trace(turn_id)
        trace.event("audio.captured", seconds=round(duration, 2), rms=round(_rms(pcm), 3))

        try:
            # The room-native path hands STT raw PCM, so it uses the provider's
            # own transcribe() instead of the codec workaround the browser
            # bridge needs for compressed webm blobs.
            with trace.span("stt", model=getattr(self.agent.provider, "stt_model", "unknown")):
                transcript = await asyncio.to_thread(
                    self.agent.provider.transcribe, pcm, SAMPLE_RATE
                )
            transcript = (transcript or "").strip()
            if not transcript:
                trace.event("stt.empty")
                return
            print(f"\ncaller> {transcript}")

            reply, action = await asyncio.to_thread(
                self.agent.respond, transcript, trace
            )
            print(f"aurora> {reply}")

            if reply:
                await self._speak(reply, turn_id=turn_id, trace=trace)

            self._write_trace(trace, action)

            if action in ("hangup", "transfer"):
                print(f"  [room] control action: {action}; leaving room")
                self._closing.set()
        except Exception as exc:
            trace.event("turn.error", errorType=type(exc).__name__, message=str(exc))
            self._write_trace(trace, None)
            print(f"  [error] {type(exc).__name__}: {exc}")

    def _write_trace(self, trace, action: str | None) -> None:
        if str(PIPELINE_ROOT) not in sys.path:
            sys.path.insert(0, str(PIPELINE_ROOT))
        from telemetry import write_trace

        write_trace(trace.finish(action=action, sources=self.agent.last_sources))

    # --- outbound audio ---

    def _is_speaking(self) -> bool:
        return self._speak_task is not None and not self._speak_task.done()

    async def _barge_in(self) -> None:
        """Stop playback the moment the caller talks over the agent.

        The browser bridge could only cancel local speech synthesis and guess
        from the transcript whether it had heard its own playback. Owning the
        published track makes this authoritative and works for any participant,
        including a future SIP caller.
        """
        self.barge_ins += 1
        self._speak_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._speak_task
        self.source.clear_queue()
        print("  [barge-in] caller interrupted; playback cancelled")

    async def _speak(self, text: str, turn_id: str, trace=None) -> None:
        """Synthesize `text` and publish it as frames on the agent track."""
        own_trace = trace is None
        trace = trace or self._trace(turn_id)
        provider = self.agent.provider
        model = getattr(provider, "tts_model", "unknown")
        backend = getattr(provider, "tts_backend", "provider")

        try:
            with trace.span("tts", model=model, backend=backend):
                wav = await asyncio.to_thread(provider.synthesize_wav, text)
                pcm = await asyncio.to_thread(wav_to_pcm, wav, SAMPLE_RATE)
        except Exception as exc:
            trace.event("tts.failed", errorType=type(exc).__name__)
            print(f"  [tts error] {type(exc).__name__}: {exc}")
            if own_trace:
                self._write_trace(trace, None)
            return

        self._speak_task = asyncio.create_task(self._publish_pcm(pcm, trace))
        with contextlib.suppress(asyncio.CancelledError):
            await self._speak_task
        if own_trace:
            self._write_trace(trace, None)

    async def _publish_pcm(self, pcm: bytes, trace) -> None:
        """Push PCM onto the track. capture_frame paces itself to real time."""
        total = len(pcm) // FRAME_BYTES
        started = time.monotonic()
        trace.event("tts.playback_started", frames=total)
        try:
            for i in range(total):
                chunk = pcm[i * FRAME_BYTES:(i + 1) * FRAME_BYTES]
                await self.source.capture_frame(
                    rtc.AudioFrame(chunk, SAMPLE_RATE, NUM_CHANNELS, FRAME_BYTES // 2)
                )
            await self.source.wait_for_playout()
            trace.event("tts.playback_complete",
                        seconds=round(time.monotonic() - started, 2))
        except asyncio.CancelledError:
            trace.event("tts.playback_interrupted",
                        seconds=round(time.monotonic() - started, 2))
            raise

    async def _shutdown(self) -> None:
        for task in (self._speak_task, self._turn_task, self._stream_task):
            if task is not None and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        await self.room.disconnect()


# --- step 1 probe, kept as the connectivity smoke test ---

class ProbeWorker:
    """Connect, subscribe to the first caller audio track, and report frames."""

    def __init__(self, room_name: str, seconds: float) -> None:
        self.room_name = room_name
        self.seconds = seconds
        self.room: rtc.Room | None = None
        self._stream_task: asyncio.Task | None = None
        self._done: asyncio.Event | None = None
        self.frames = 0
        self.bytes_in = 0
        self.peak_rms = 0.0

    async def run(self) -> int:
        self.room = rtc.Room()
        self._done = asyncio.Event()

        @self.room.on("track_subscribed")
        def _on_track(track, publication, participant) -> None:
            if track.kind != rtc.TrackKind.KIND_AUDIO or self._stream_task is not None:
                return
            print(f"  [room] subscribed to audio from {participant.identity!r}")
            self._stream_task = asyncio.create_task(self._consume(track))

        url = _livekit_url()
        print(f"Connecting to {url} as {AGENT_IDENTITY!r} in room {self.room_name!r}")
        await self.room.connect(url, _token(AGENT_IDENTITY, AGENT_NAME, self.room_name))
        print(f"Listening for caller audio for {self.seconds:.0f}s.")

        for participant in self.room.remote_participants.values():
            for publication in participant.track_publications.values():
                if publication.track is not None and self._stream_task is None:
                    self._stream_task = asyncio.create_task(
                        self._consume(publication.track)
                    )

        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self._done.wait(), timeout=self.seconds)

        if self._stream_task is not None:
            self._stream_task.cancel()
            await asyncio.gather(self._stream_task, return_exceptions=True)
        await self.room.disconnect()
        return self._report()

    async def _consume(self, track: rtc.Track) -> None:
        stream = rtc.AudioStream(
            track,
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
            frame_size_ms=FRAME_MS,
        )
        try:
            async for event in stream:
                pcm = bytes(event.frame.data)
                self.frames += 1
                self.bytes_in += len(pcm)
                self.peak_rms = max(self.peak_rms, _rms(pcm))
                if self.frames == 1:
                    print(
                        f"  [audio] first frame: {event.frame.sample_rate} Hz, "
                        f"{event.frame.num_channels}ch, "
                        f"{event.frame.samples_per_channel} samples, {len(pcm)} bytes"
                    )
        except asyncio.CancelledError:
            pass
        finally:
            await stream.aclose()

    def _report(self) -> int:
        print("\n--- probe result ---")
        print(f"frames received : {self.frames}")
        print(f"audio duration  : {self.frames * FRAME_MS / 1000:.1f}s")
        print(f"peak RMS        : {self.peak_rms:.3f}")
        if self.frames == 0:
            print("FAIL: no audio frames. Is a participant publishing a mic track?")
            return 1
        if self.bytes_in / self.frames != FRAME_BYTES:
            print(f"FAIL: expected {FRAME_BYTES} bytes/frame for 16kHz mono 20ms.")
            return 1
        if self.peak_rms < 0.001:
            print("WARN: frames arrived but are silent. Check mic permission.")
            return 1
        print("PASS: room-native audio reaches Python at 16kHz mono.")
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true",
                        help="Report inbound audio and exit without answering.")
    parser.add_argument("--seconds", type=float, default=15.0,
                        help="Probe listen duration.")
    parser.add_argument("--room", default=None, help="Room name override.")
    parser.add_argument("--provider", default=None,
                        help="Override PROVIDER (mock, groq, openai).")
    parser.add_argument("--no-greeting", action="store_true",
                        help="Do not speak the opening line on connect.")
    args = parser.parse_args()

    load_env_files((PIPELINE_ROOT / ".env", ROOT / ".env"))
    os.environ.setdefault(
        "TELEMETRY_JSONL",
        str(ASSIGNMENT_ROOT / "logs" / "voice-events.jsonl"),
    )
    room_name = args.room or os.getenv("LIVEKIT_ROOM", "aurora-demo-room")
    provider_name = (args.provider or os.getenv("PROVIDER", "mock")).lower()

    worker = (
        ProbeWorker(room_name, args.seconds)
        if args.probe
        else AuroraWorker(room_name, provider_name, greet=not args.no_greeting)
    )
    try:
        sys.exit(asyncio.run(worker.run()))
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
