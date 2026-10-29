"""0.19.5: the per-generation speaker check (0.19.4) must not starve the listener. Weight-free.

A fake runtime generates 0.96 s chunks on a VIRTUAL clock at a chosen real-time
factor; the listener plays in real time from the first chunk it receives. The
reply must play without gaps, and keep the one-voice rule.
"""
from __future__ import annotations

import numpy as np
import pytest

from abstractvoice.cloning.engine_qwen3_tts import ClonedVoiceNotAppliedError, Qwen3TTSVoiceCloningEngine
from abstractvoice.qwen3_tts.runtime import Qwen3TTSSettings

SR = 24000
STEP = int(0.96 * SR)
FIRST = int(0.475 * SR)  # the MLX stream's first chunk (6 codec frames, minus the 5 ms fade hold)
CLONE, STOCK = 0.98, 0.90


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class _RT:
    """Native-stream fake whose generation costs ``rtf`` seconds of virtual time per second of audio."""

    def __init__(self, clock, *, rtf, seconds, stock_segments=()):
        self.settings = Qwen3TTSSettings()
        self.supports_native_streaming = True
        self.clock, self.rtf, self.seconds, self.stock = clock, rtf, seconds, set(stock_segments)
        self.texts = []

    def build_clone_prompt(self, *, ref_audio, ref_text, x_vector_only):
        return "prompt"

    def stream_clone(self, text, *, clone_prompt, language=None, **kw):
        if text not in self.texts:
            self.texts.append(text)
        seg = self.texts.index(text)
        n, pos, first = int(self.seconds * SR), 0, True
        while pos < n:
            size = min(FIRST if first else STEP, n - pos)
            first = False
            self.clock.t += self.rtf * size / SR
            value = (STOCK if seg in self.stock else CLONE) + 0.0001 * seg  # the sample value encodes the voice
            yield np.full(size, value, dtype=np.float32), SR
            pos += size

    def speaker_similarity(self, audio, *, clone_prompt, sample_rate=SR):
        return float(np.round(np.asarray(audio).reshape(-1).min(), 2))


def _play(engine, text):
    """Arrival times and durations; then the listener's gaps (playing from the first arrival on)."""
    arrivals = []
    for audio, sr in engine.infer_to_audio_chunks(text=text, reference_paths=["ref.wav"], reference_text="t"):
        arrivals.append((engine._clock(), audio.size / sr, float(audio[0])))
    end, gaps = None, []
    for t, d, _ in arrivals:
        if end is None:
            end = t
        if t > end + 1e-9:
            gaps.append(t - end)
        end = max(end, t) + d
    return arrivals, gaps


def _engine(rtf, seconds=20.0, stock_segments=()):
    clock = _Clock()
    engine = Qwen3TTSVoiceCloningEngine(allow_downloads=False)
    engine._clock = clock
    engine._runtime = _RT(clock, rtf=rtf, seconds=seconds, stock_segments=stock_segments)
    return engine


# Three ~800-char paragraphs: three generations (segments) of 20 s each, a 60 s reply.
TEXT = " ".join(
    " ".join(f"Paragraph {p} sentence {i} says something long enough to be spoken." for i in range(12))
    for p in range(3)
)


@pytest.mark.parametrize("rtf", [0.2, 0.6, 0.9])
def test_a_long_reply_plays_without_gaps_at_any_realtime_factor(rtf):
    engine = _engine(rtf)
    arrivals, gaps = _play(engine, TEXT)
    assert len(engine.last_segments) == 3
    assert sum(d for _, d, _ in arrivals) == pytest.approx(60.0, abs=0.1)
    assert max(gaps, default=0.0) < 0.1, f"listener starved: gaps {[round(g, 2) for g in gaps]}"
    # Only the reply's first opening pays the check: first audio within its 2-4 s opening.
    assert arrivals[0][0] <= 4.4 * rtf + 1e-6


def test_delivery_follows_the_generation_cadence_not_held_blocks():
    """0.19.4 released each verified 4 s block at once; a listener-side player fed in lumps cuts."""
    engine = _engine(0.6)
    arrivals, _ = _play(engine, TEXT)
    # After the first opening and before the reply's last window (flushed once the reply is generated).
    after_opening = [t for t, _, _ in arrivals[5:-6]]
    same_instant = max(sum(1 for u in after_opening if abs(u - t) < 1e-9) for t in after_opening)
    assert same_instant <= 2, f"{same_instant} chunks released at the same instant"


def test_at_ordinary_speed_every_chunk_is_verified_before_it_plays():
    engine = _engine(0.2)
    _play(engine, TEXT)
    assert all(r["provisional_s"] == 0 for r in engine.last_segments)
    assert all(r["status"] == "ok" for r in engine.last_segments)


def test_a_stock_voice_segment_still_never_plays_and_the_reply_stops_naming_it():
    engine = _engine(0.6, stock_segments={1})
    delivered = []
    with pytest.raises(ClonedVoiceNotAppliedError) as info:
        for audio, _sr in engine.infer_to_audio_chunks(text=TEXT, reference_paths=["ref.wav"], reference_text="t"):
            delivered.append(float(audio[0]))
    assert info.value.segment_index == 1
    assert delivered and all(v >= CLONE for v in delivered)  # segment 0 in full, nothing of segment 1
    assert sum(1 for _ in delivered) > 15  # segment 0 was delivered completely before the stop


def test_near_realtime_the_listener_is_fed_and_a_voice_change_still_stops_the_reply():
    # At RTF 1.0 the check cannot run ahead: chunks go out before their window is scored,
    # the reply is still stopped as soon as a score shows another voice.
    engine = _engine(1.0, stock_segments={1})
    delivered, gaps = [], []
    end = None
    with pytest.raises(ClonedVoiceNotAppliedError) as info:
        for audio, sr in engine.infer_to_audio_chunks(text=TEXT, reference_paths=["ref.wav"], reference_text="t"):
            t = engine._clock()
            if end is not None and t > end + 1e-9:
                gaps.append(t - end)
            end = (t if end is None else max(end, t)) + audio.size / sr
            delivered.append(float(audio[0]))
    assert info.value.segment_index == 1 and info.value.mid_segment
    assert max(gaps, default=0.0) < 0.1
    stock_heard = sum(1 for v in delivered if v < CLONE) * 0.96
    assert stock_heard <= 4.0  # at most the opening window before the stop
