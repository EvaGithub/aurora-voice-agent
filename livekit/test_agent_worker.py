"""Unit tests for the room-native worker's audio logic.

These cover the parts that decide when a caller turn is complete and what gets
published back, none of which need a LiveKit connection. Room behaviour itself
is exercised by `sim_caller.py` against a running server.

    python -m unittest -v test_agent_worker.py
"""

from __future__ import annotations

import array
import io
import math
import sys
import unittest
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

import agent_worker as W


def tone_frame(samples: int = 320, amplitude: int = 8000) -> bytes:
    """One frame of a sine tone, which webrtcvad classifies as speech."""
    return array.array(
        "h", [int(amplitude * math.sin(i * 0.35)) for i in range(samples)]
    ).tobytes()


def silence_frame(samples: int = 320) -> bytes:
    return b"\x00" * samples * 2


def make_wav(pcm: bytes, rate: int, channels: int = 1, width: int = 2) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


class FrameGeometryTest(unittest.TestCase):
    """The frame size must line up with what webrtcvad accepts."""

    def test_frame_bytes_match_a_single_vad_window(self):
        # 16 kHz, 16-bit, 20 ms -> 320 samples -> 640 bytes.
        self.assertEqual(W.FRAME_BYTES, 640)
        self.assertEqual(W.FRAME_BYTES // 2, W.SAMPLE_RATE * W.FRAME_MS // 1000)

    def test_vad_accepts_the_configured_frame_size(self):
        import webrtcvad

        vad = webrtcvad.Vad(2)
        # Raises if the frame size is not 10/20/30 ms at a supported rate.
        self.assertIsInstance(vad.is_speech(tone_frame(), W.SAMPLE_RATE), bool)


class EndpointerTest(unittest.TestCase):
    def setUp(self):
        self.endpointer = W.Endpointer(aggressiveness=2, silence_ms=600)

    def _feed(self, frame: bytes, count: int) -> list[bytes]:
        return [out for _ in range(count) if (out := self.endpointer.push(frame))]

    def test_silence_alone_never_commits_a_turn(self):
        self.assertEqual(self._feed(silence_frame(), 100), [])

    def test_speech_then_silence_commits_exactly_one_turn(self):
        self.assertEqual(self._feed(tone_frame(), 60), [])
        committed = self._feed(silence_frame(), 40)
        self.assertEqual(len(committed), 1)

    def test_committed_audio_includes_preroll_so_onset_is_not_clipped(self):
        """Onset needs several frames to confirm; those must not be discarded."""
        self._feed(tone_frame(), 60)
        pcm = self._feed(silence_frame(), 40)[0]
        frames = len(pcm) // W.FRAME_BYTES
        # 60 speech frames arrive, but onset only confirms after 3, so without
        # the pre-roll buffer the first frames would be lost.
        self.assertGreater(frames, 60)

    def test_short_blip_is_rejected_as_noise(self):
        self._feed(tone_frame(), 4)  # ~80ms, under MIN_UTTERANCE_MS
        self.assertEqual(self._feed(silence_frame(), 40), [])

    def test_endpointer_resets_and_can_take_a_second_turn(self):
        self._feed(tone_frame(), 60)
        self.assertEqual(len(self._feed(silence_frame(), 40)), 1)
        self.assertFalse(self.endpointer.capturing)
        self._feed(tone_frame(), 60)
        self.assertEqual(len(self._feed(silence_frame(), 40)), 1)

    def test_brief_pause_mid_sentence_does_not_end_the_turn(self):
        self._feed(tone_frame(), 60)
        # 300ms of silence is a pause, not an endpoint (threshold is 600ms).
        self.assertEqual(self._feed(silence_frame(), 15), [])
        self.assertTrue(self.endpointer.capturing)


class WavToPcmTest(unittest.TestCase):
    def test_passthrough_when_rate_already_matches(self):
        pcm = tone_frame(1600)
        out = W.wav_to_pcm(make_wav(pcm, 16000), 16000)
        self.assertEqual(out, pcm)

    def test_downsamples_provider_audio_to_the_published_rate(self):
        pcm = tone_frame(2400)
        out = W.wav_to_pcm(make_wav(pcm, 24000), 16000)
        self.assertAlmostEqual(len(out) / len(pcm), 2 / 3, places=2)

    def test_upsamples_low_rate_audio(self):
        pcm = tone_frame(800)
        out = W.wav_to_pcm(make_wav(pcm, 8000), 16000)
        self.assertAlmostEqual(len(out) / len(pcm), 2.0, places=2)

    def test_stereo_is_mixed_down_to_mono(self):
        mono = array.array("h", [100, 200, 300, 400])
        stereo = array.array("h", [100, 100, 200, 200, 300, 300, 400, 400])
        out = W.wav_to_pcm(make_wav(stereo.tobytes(), 16000, channels=2), 16000)
        self.assertEqual(out, mono.tobytes())

    def test_rejects_non_16_bit_audio(self):
        with self.assertRaises(ValueError):
            W.wav_to_pcm(make_wav(b"\x00" * 100, 16000, width=1), 16000)

    def test_output_length_is_always_frame_aligned(self):
        out = W.wav_to_pcm(make_wav(tone_frame(2400), 24000), 16000)
        self.assertEqual(len(out) % 2, 0)


class SystemTtsBytesTest(unittest.TestCase):
    """The worker cannot publish audio the system voice played to speakers."""

    def test_synthesize_wav_returns_publishable_bytes_in_mock_mode(self):
        from providers import MockProvider

        wav = MockProvider().synthesize_wav("Testing one two three.")
        self.assertGreater(len(wav), 1000)
        pcm = W.wav_to_pcm(wav, W.SAMPLE_RATE)
        self.assertGreater(len(pcm), W.FRAME_BYTES)

    def test_plain_synthesize_still_returns_none_for_the_existing_callers(self):
        """voice_loop and talk_server rely on the original contract."""
        from providers import MockProvider

        provider = MockProvider()
        provider.tts_backend = "print"
        self.assertIsNone(provider.synthesize("unchanged"))


if __name__ == "__main__":
    unittest.main()
