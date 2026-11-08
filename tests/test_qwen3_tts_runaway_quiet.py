"""A Qwen3-TTS generation that misses its end of speech is ended where it went quiet (weight-free).

2026-11-08, live: a 619-character clone segment ran to its 2048-frame cap (163.84 s). Its speech
stopped at 25 s; the listener heard a repeated syllable, then 2 minutes of near-silence with stray
bursts, then the next segment. The runtime now ends a generation once 90 % of its last 4 s are
quiet (real speech: at most 34 % over 52 real takes), and says so: ``report["ended"]`` and the
clone engine's segment record.
"""
from __future__ import annotations

import numpy as np
import pytest

from abstractvoice.cloning.engine_qwen3_tts import Qwen3TTSVoiceCloningEngine
from abstractvoice.qwen3_tts import runtime as runtime_module
from abstractvoice.qwen3_tts.runtime import (
    RUNAWAY_WINDOW_SECONDS,
    Qwen3TTSRuntime,
    _QuietRunaway,
    estimate_max_new_tokens,
)

SR = 24000
FRAME = SR * 2 // 25  # one 12.5 Hz codec frame: 1920 samples
TEXT = "x" * 619  # the live segment's length: a 2048-frame cap, 163.84 s


def _speech(seconds: float, *, talk: float = 2.0, pause: float = 1.5) -> np.ndarray:
    """Voiced audio (-17 dBFS tone) with a real pause of ``pause`` s after every ``talk`` s."""
    t = np.arange(int(seconds * SR)) / SR
    audio = (0.2 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
    audio[(t % (talk + pause)) >= talk] = 0.0
    return audio


def _runaway_tail(seconds: float, seed: int = 0) -> np.ndarray:
    """Near-silence (-50 dBFS noise) with a 0.3 s burst every 7 s: the live runaway after its speech."""
    rng = np.random.default_rng(seed)
    audio = (rng.standard_normal(int(seconds * SR)) * 0.003).astype(np.float32)
    burst = (0.2 * np.sin(2 * np.pi * 200 * np.arange(int(0.3 * SR)) / SR)).astype(np.float32)
    for at in np.arange(1.0, seconds - 0.3, 7.0):
        audio[int(at * SR):int(at * SR) + burst.size] = burst
    return audio


def _chunks(audio: np.ndarray, size: int):
    return [audio[i:i + size] for i in range(0, audio.size, size)]


def _push_all(audio: np.ndarray, size: int):
    """(samples kept before the cut, quiet_from) -- or (all samples, None) when it never cuts."""
    watch = _QuietRunaway(SR)
    kept = 0
    for chunk in _chunks(audio, size):
        keep = watch.push(chunk)
        if keep is not None:
            return kept + keep, watch.quiet_from
        kept += chunk.size
    return kept, None


# ------------------------------------------------------------------------------ the watch


@pytest.mark.parametrize("size", [FRAME, 12 * FRAME, 10**7], ids=["frame", "chunk", "whole"])
def test_speech_with_real_pauses_is_never_cut(size):
    audio = np.concatenate([np.zeros(SR, np.float32), _speech(160.0), np.zeros(SR // 2, np.float32)])
    assert _push_all(audio, size) == (audio.size, None)


@pytest.mark.parametrize("size", [FRAME, 12 * FRAME, 10**7], ids=["frame", "chunk", "whole"])
def test_a_generation_that_goes_quiet_is_cut_where_its_quiet_window_began(size):
    audio = np.concatenate([_speech(25.0, pause=0.6), _runaway_tail(138.84)])
    kept, quiet_from = _push_all(audio, size)
    # The window may start a little before the speech ends (it needs only 90 % quiet).
    assert quiet_from is not None and 24.0 <= quiet_from / SR <= 25.0 + 2 * RUNAWAY_WINDOW_SECONDS
    # Streamed, the audio already sent stays sent: at most the quiet window plus one chunk is heard.
    assert kept / SR <= quiet_from / SR + RUNAWAY_WINDOW_SECONDS + size / SR
    if size == 10**7:  # a whole (PyTorch) take is cut right where it went quiet
        assert kept == quiet_from


def test_leading_silence_does_not_count():
    audio = np.concatenate([np.zeros(6 * SR, np.float32), _speech(10.0)])
    assert _push_all(audio, 12 * FRAME) == (audio.size, None)


# --------------------------------------------------------------------- the MLX generation


class _FakeMLX:
    """A model whose frame stream is ``audio`` (cut at ``max_new_tokens`` frames), closable."""

    sample_rate = SR

    def __init__(self, audio: np.ndarray):
        self.audio = audio
        self.closed = False
        self.frames_made = 0

    def stream(self, prompt, sampling, *, max_new_tokens, chunk_frames, first_chunk_frames):
        audio = self.audio[: max_new_tokens * FRAME]
        try:
            for start in range(0, audio.size, chunk_frames * FRAME):
                chunk = audio[start:start + chunk_frames * FRAME]
                self.frames_made += chunk.size // FRAME
                yield chunk
        finally:
            self.closed = True


def _runtime(audio: np.ndarray, monkeypatch) -> Qwen3TTSRuntime:
    rt = Qwen3TTSRuntime(allow_downloads=False)
    rt._mlx = _FakeMLX(audio)
    rt._backend = "mlx"
    monkeypatch.setattr(rt, "_sampling_params", lambda overrides: None)
    return rt


def _speak(rt: Qwen3TTSRuntime):
    report: dict = {}
    out = list(rt._mlx_stream(TEXT, lambda model: "prompt", {}, report=report))
    return sum(a.size for a, _ in out) / SR, report


def test_the_live_runaway_ends_about_four_seconds_after_its_quiet_began(monkeypatch):
    assert estimate_max_new_tokens(TEXT) == 2048
    rt = _runtime(np.concatenate([_speech(25.0, pause=0.6), _runaway_tail(200.0)]), monkeypatch)
    heard, report = _speak(rt)
    assert report["ended"] == "quiet_runaway" and 24.0 <= report["quiet_from_s"] <= 33.0
    assert heard <= report["quiet_from_s"] + RUNAWAY_WINDOW_SECONDS + 1.0  # was 163.84 s
    assert report["audio_s"] == pytest.approx(heard, abs=0.01)  # what was delivered, not what was made
    assert rt._mlx.closed and rt._mlx.frames_made < 2048  # the generation itself stopped
    assert rt._lock.acquire(timeout=0.1)  # and released the model


def test_the_check_goes_red_without_the_guard(monkeypatch):
    # Mutant: a fraction no window can reach. The same take then runs to its cap, as it did live.
    monkeypatch.setattr(runtime_module, "RUNAWAY_QUIET_FRACTION", 1.01)
    rt = _runtime(np.concatenate([_speech(25.0, pause=0.6), _runaway_tail(200.0)]), monkeypatch)
    heard, report = _speak(rt)
    assert report["ended"] == "cap" and heard == pytest.approx(163.84, abs=0.01)


def test_a_generation_that_ends_at_its_eos_reports_eos(monkeypatch):
    heard, report = _speak(_runtime(_speech(40.0), monkeypatch))
    assert report == {"ended": "eos", "audio_s": 40.0, "max_new_tokens": 2048} and heard == pytest.approx(40.0)


def test_speech_up_to_the_cap_reports_cap(monkeypatch):
    heard, report = _speak(_runtime(_speech(200.0), monkeypatch))
    assert report["ended"] == "cap" and heard == pytest.approx(163.84, abs=0.01)


def test_a_whole_pytorch_take_is_cut_where_it_went_quiet():
    rt = Qwen3TTSRuntime(allow_downloads=False)
    report: dict = {}
    audio, sr = rt._finish_generation(
        np.concatenate([_speech(25.0, pause=0.6), _runaway_tail(138.84)]), SR, TEXT, 2048, report=report
    )
    assert report["ended"] == "quiet_runaway" and audio.size / sr == pytest.approx(report["quiet_from_s"], abs=0.01)


# ------------------------------------------------------------------ the clone segment record


class _CloneRuntime:
    supports_native_streaming = True

    def build_clone_prompt(self, **kwargs):
        return "prompt"

    def stream_clone(self, text, *, clone_prompt, language=None, report=None):
        yield _speech(6.0), SR
        report.update({"ended": "quiet_runaway", "audio_s": 10.0, "max_new_tokens": 2048, "quiet_from_s": 6.0})

    def speaker_similarity(self, audio, *, clone_prompt, sample_rate=SR):
        return 0.99


def test_the_segment_record_says_how_its_generation_ended(tmp_path):
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"RIFF")
    engine = Qwen3TTSVoiceCloningEngine(allow_downloads=False)
    engine._runtime = _CloneRuntime()
    chunks = list(engine.infer_to_audio_chunks(text="One sentence.", reference_paths=[str(ref)], reference_text="Ref"))
    assert chunks
    record = engine.last_segments[0]
    assert record["status"] == "ok" and record["ended"] == "quiet_runaway" and record["quiet_from_s"] == 6.0
