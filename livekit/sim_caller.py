"""Simulated caller: publish a WAV file into the room as a real audio track.

The room-native worker needs a participant publishing microphone audio. Doing
that by hand means a browser, a microphone, and a person talking, which is not
repeatable. This script joins as `caller` and publishes speech synthesized by
the local system voice, so the worker can be exercised end to end from a script.

    python sim_caller.py --say "What is the cancellation policy?"
    python sim_caller.py --wav some_recording.wav
"""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import tempfile
import wave
import warnings
from pathlib import Path

import jwt
from livekit import api, rtc

from env_loader import load_env_files

ROOT = Path(__file__).resolve().parent
PIPELINE_ROOT = ROOT.parent / "pipeline"

FRAME_MS = 20
CALLER_IDENTITY = os.getenv("CALLER_IDENTITY", "caller")


def _livekit_url() -> str:
    raw = os.getenv("LIVEKIT_URL", "ws://localhost:7880")
    if raw.startswith("http://"):
        return "ws://" + raw[len("http://"):]
    if raw.startswith("https://"):
        return "wss://" + raw[len("https://"):]
    return raw


def _token(identity: str, room: str) -> str:
    secret = os.getenv("LIVEKIT_API_SECRET", "secret")
    if secret == "secret":
        warnings.filterwarnings("ignore", category=jwt.InsecureKeyLengthWarning)
    return (
        api.AccessToken(os.getenv("LIVEKIT_API_KEY", "devkey"), secret)
        .with_identity(identity)
        .with_name(identity)
        .with_grants(
            api.VideoGrants(room_join=True, room=room, can_publish=True, can_subscribe=True)
        )
        .to_jwt()
    )


def synth_wav(text: str, path: Path, rate: int = 16000) -> Path:
    """Render `text` to a 16-bit mono WAV with the macOS system voice.

    `say` writes a real file here rather than playing to the speakers, which is
    the same trick the worker uses to get TTS bytes it can publish.
    """
    subprocess.run(
        ["say", "--data-format=LEI16@%d" % rate, "-o", str(path), text],
        check=True,
    )
    return path


def read_wav(path: Path) -> tuple[bytes, int]:
    with wave.open(str(path), "rb") as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError("Expected 16-bit mono WAV")
        return w.readframes(w.getnframes()), w.getframerate()


class AgentListener:
    """Subscribe to the agent's published track and measure what comes back.

    Publishing audio is only half of room-native operation. Without this, a
    worker could look correct while sending its reply nowhere. Listening from
    the caller side proves the full round trip through the room.
    """

    def __init__(self) -> None:
        self.frames = 0
        self.peak_rms = 0.0
        self.task: asyncio.Task | None = None
        self.pcm = bytearray()

    def attach(self, room: rtc.Room, agent_identity: str) -> None:
        @room.on("track_subscribed")
        def _on_track(track, publication, participant) -> None:
            if track.kind != rtc.TrackKind.KIND_AUDIO:
                return
            if participant.identity != agent_identity or self.task is not None:
                return
            print(f"caller subscribed to {participant.identity!r} audio")
            self.task = asyncio.create_task(self._consume(track))

    async def _consume(self, track: rtc.Track) -> None:
        import array
        import math

        stream = rtc.AudioStream(track, sample_rate=16000, num_channels=1,
                                 frame_size_ms=FRAME_MS)
        try:
            async for event in stream:
                data = bytes(event.frame.data)
                self.frames += 1
                self.pcm.extend(data)
                samples = array.array("h")
                samples.frombytes(data)
                if samples:
                    rms = math.sqrt(sum(s * s for s in samples) / len(samples)) / 32768
                    self.peak_rms = max(self.peak_rms, rms)
        except asyncio.CancelledError:
            pass
        finally:
            await stream.aclose()

    def report(self, save: str | None = None) -> int:
        seconds = self.frames * FRAME_MS / 1000
        print("\n--- agent audio received by caller ---")
        print(f"frames   : {self.frames}")
        print(f"duration : {seconds:.1f}s")
        print(f"peak RMS : {self.peak_rms:.3f}")
        if save and self.pcm:
            with wave.open(save, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(16000)
                w.writeframes(bytes(self.pcm))
            print(f"saved    : {save}")
        if self.frames == 0:
            print("FAIL: the agent published no audio into the room.")
            return 1
        if self.peak_rms < 0.001:
            print("FAIL: agent track carried only silence.")
            return 1
        print("PASS: agent speech travelled back through the LiveKit room.")
        return 0


async def publish(pcm: bytes, rate: int, room_name: str, hold: float,
                  listen: float = 0.0, save: str | None = None,
                  interrupt_pcm: bytes | None = None,
                  interrupt_after: float = 2.0) -> int:
    room = rtc.Room()
    listener = AgentListener()
    if listen:
        listener.attach(room, os.getenv("AGENT_IDENTITY", "aurora"))
    await room.connect(_livekit_url(), _token(CALLER_IDENTITY, room_name))
    print(f"caller connected to {room_name!r} as {CALLER_IDENTITY!r}")

    source = rtc.AudioSource(rate, 1)
    track = rtc.LocalAudioTrack.create_audio_track("caller-mic", source)
    options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
    await room.local_participant.publish_track(track, options)
    print("caller published mic track")

    # Give the worker a moment to subscribe before the speech starts, otherwise
    # the opening words land before the stream is attached.
    await asyncio.sleep(1.0)

    samples_per_frame = int(rate * FRAME_MS / 1000)
    bytes_per_frame = samples_per_frame * 2
    silence = b"\x00" * bytes_per_frame

    async def speak(buf: bytes, label: str) -> None:
        total = len(buf) // bytes_per_frame
        print(f"caller {label}: {total} frames ({total * FRAME_MS / 1000:.1f}s)")
        for i in range(total):
            await source.capture_frame(
                rtc.AudioFrame(
                    buf[i * bytes_per_frame:(i + 1) * bytes_per_frame],
                    rate, 1, samples_per_frame,
                )
            )

    async def quiet(seconds: float) -> None:
        # Real silence keeps the track live so the worker's endpointer sees a
        # pause rather than the track simply disappearing.
        for _ in range(int(seconds * 1000 / FRAME_MS)):
            await source.capture_frame(
                rtc.AudioFrame(silence, rate, 1, samples_per_frame)
            )

    await speak(pcm, "speaking")
    await quiet(hold)

    if interrupt_pcm is not None:
        print(f"caller waiting {interrupt_after:.1f}s, then talking over the agent")
        await quiet(interrupt_after)
        await speak(interrupt_pcm, "INTERRUPTING")
        await quiet(hold)

    print("caller finished speaking")

    status = 0
    if listen:
        print(f"caller listening for the agent reply ({listen:.0f}s)...")
        await asyncio.sleep(listen)
        status = listener.report(save)
        if listener.task is not None:
            listener.task.cancel()
            await asyncio.gather(listener.task, return_exceptions=True)

    await room.disconnect()
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--say", default="What is the cancellation policy?",
                        help="Text to speak with the system voice.")
    parser.add_argument("--wav", default=None, help="Publish this WAV instead.")
    parser.add_argument("--room", default=None, help="Room name override.")
    parser.add_argument("--hold", type=float, default=2.0,
                        help="Seconds of trailing silence after speech.")
    parser.add_argument("--listen", type=float, default=0.0,
                        help="Seconds to listen for the agent's reply, then verify it.")
    parser.add_argument("--save", default=None,
                        help="Write the received agent audio to this WAV path.")
    parser.add_argument("--interrupt", default=None,
                        help="Speak this over the agent's reply to test barge-in.")
    parser.add_argument("--interrupt-after", type=float, default=2.0,
                        help="Seconds into the agent's reply before interrupting.")
    args = parser.parse_args()

    load_env_files((PIPELINE_ROOT / ".env", ROOT / ".env"))
    room_name = args.room or os.getenv("LIVEKIT_ROOM", "aurora-demo-room")

    interrupt_pcm = None
    with tempfile.TemporaryDirectory() as tmp:
        if args.wav:
            pcm, rate = read_wav(Path(args.wav))
        else:
            pcm, rate = read_wav(synth_wav(args.say, Path(tmp) / "caller.wav"))
        if args.interrupt:
            interrupt_pcm, _ = read_wav(
                synth_wav(args.interrupt, Path(tmp) / "interrupt.wav", rate)
            )

    raise SystemExit(
        asyncio.run(publish(pcm, rate, room_name, args.hold, args.listen,
                            args.save, interrupt_pcm, args.interrupt_after))
    )


if __name__ == "__main__":
    main()
