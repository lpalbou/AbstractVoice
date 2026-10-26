"""Qwen3-TTS on MLX, end to end on the real checkpoints (ADR 0008).

Gated: set ``ABSTRACTVOICE_RUN_QWEN3_TTS_MLX_TESTS=1`` on an Apple silicon Mac with ``mlx``
installed and the snapshots prefetched (``abstractvoice-prefetch --qwen3-tts`` for the 0.6B
CustomVoice, plus ``Qwen/Qwen3-TTS-12Hz-0.6B-Base`` and ``Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign``).
Marked ``model_download`` so CI's default run skips it; elsewhere every test skips cleanly.

Latency and real-time-factor assertions are deliberately looser than the figures reported in
``docs/backlog/completed/0061_qwen3_tts_mlx_backend.md``: the owner's laptop runs other work.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

from abstractvoice.qwen3_tts.backend import mlx_unavailable_reason

pytestmark = [
    pytest.mark.model_download,
    pytest.mark.skipif(
        os.environ.get("ABSTRACTVOICE_RUN_QWEN3_TTS_MLX_TESTS", "0") != "1",
        reason="set ABSTRACTVOICE_RUN_QWEN3_TTS_MLX_TESTS=1 to run the Qwen3-TTS MLX end-to-end tests",
    ),
    pytest.mark.skipif(mlx_unavailable_reason() is not None, reason=str(mlx_unavailable_reason())),
]

CUSTOM = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
BASE = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
DESIGN = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
REFERENCE_TEXT = "This reference voice belongs to the abstract framework test suite."
REPLY = (
    "Streaming speech is working. The second sentence keeps the same voice, because it is the same "
    "generation. And a third sentence confirms that nothing is re-sampled between them."
)


def _cached(model_id: str) -> bool:
    from abstractvoice.local_models import hf_cached_snapshot_dir

    return hf_cached_snapshot_dir(model_id) is not None


def _words(text: str) -> list[str]:
    return "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in text).split()


def _wer(reference: str, hypothesis: str) -> float:
    ref, hyp = _words(reference), _words(hypothesis)
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
            prev, d[j] = d[j], cur
    return d[len(hyp)] / max(1, len(ref))


def _transcribe(audio: np.ndarray, sr: int) -> str:
    """faster-whisper large-v3-turbo on the CPU (skips when it is not installed/cached)."""
    fw = pytest.importorskip("faster_whisper")
    try:
        model = fw.WhisperModel("large-v3-turbo", device="cpu", compute_type="int8")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"faster-whisper large-v3-turbo is not available: {e}")
    segments, _ = model.transcribe(audio.astype(np.float32), language="en", beam_size=1)
    return " ".join(s.text.strip() for s in segments)


@pytest.fixture(scope="module")
def custom_runtime():
    if not _cached(CUSTOM):
        pytest.skip(f"{CUSTOM} is not in the local Hugging Face cache")
    from abstractvoice.qwen3_tts.runtime import Qwen3TTSRuntime

    runtime = Qwen3TTSRuntime(CUSTOM, allow_downloads=False)
    assert runtime.resolve_backend() == "mlx"
    t0 = time.perf_counter()
    runtime._ensure_loaded()
    runtime.load_plus_prime_s = time.perf_counter() - t0  # type: ignore[attr-defined]
    yield runtime
    runtime.unload()


def test_custom_voice_loads_primes_and_reports_the_backend(custom_runtime):
    info = custom_runtime.runtime_info()
    assert info["backend"] == "mlx" and info["device"] == "metal" and info["native_streaming"] is True
    assert info["mlx_quantization"] in ("bf16", "8bit", "4bit")
    assert info["pytorch_switches"]["applied"] is False
    assert info["prime_seconds"] is not None and info["prime_seconds"] > 0
    assert custom_runtime.load_plus_prime_s < 30.0
    # Speaker presets are identical to the weight-free (config) listing the PyTorch path uses.
    from abstractvoice.qwen3_tts.runtime import Qwen3TTSRuntime

    assert custom_runtime.speaker_names() == Qwen3TTSRuntime(CUSTOM, allow_downloads=False).speaker_names()
    assert len(custom_runtime.speaker_names()) == 9


def test_streamed_reply_is_one_generation_of_contiguous_frames(custom_runtime):
    t0 = time.perf_counter()
    first = None
    chunks = []
    for audio, sr in custom_runtime.stream_custom_voice(REPLY, speaker="ryan", language="English"):
        if first is None:
            first = time.perf_counter() - t0
        chunks.append(np.asarray(audio))
        assert sr == 24000
    wall = time.perf_counter() - t0
    audio = np.concatenate(chunks)
    seconds = len(audio) / 24000
    assert first is not None and first < 2.0, f"first chunk after {first:.2f}s"
    assert seconds > 5.0 and wall / seconds < 1.0, f"RTF {wall / seconds:.2f}"
    assert len(chunks) >= 4
    # Contiguous frames: chunk sizes are whole codec frames (1920 samples) except the 5 ms tail,
    # only the utterance edges are faded, and the seams are not.
    sizes = [len(c) for c in chunks]
    fade = int(round(24000 * 5 / 1000))
    assert sizes[0] == 6 * 1920 - fade and sizes[-1] == fade and all(s % 1920 == 0 for s in sizes[1:-1])
    assert audio[0] == 0.0 and audio[-1] == 0.0
    seam = chunks[0][-50:]
    assert np.abs(seam).max() > 0.0  # no fade-out at a seam
    custom_runtime.first_chunk_s = first  # type: ignore[attr-defined]


def test_buffered_and_streamed_outputs_transcribe_to_the_text(custom_runtime):
    audio, sr = custom_runtime.synthesize_custom_voice(REPLY, speaker="ryan", language="English")
    assert sr == 24000 and len(audio) > 24000 * 5
    wer = _wer(REPLY, _transcribe(audio, sr))
    assert wer < 0.2, f"WER {wer:.2f}"


def test_voice_design_streams_on_mlx():
    if not _cached(DESIGN):
        pytest.skip(f"{DESIGN} is not in the local Hugging Face cache")
    from abstractvoice.qwen3_tts.runtime import Qwen3TTSRuntime

    runtime = Qwen3TTSRuntime(DESIGN, allow_downloads=False)
    try:
        chunks = list(
            runtime.stream_voice_design(
                "This voice was described in words, and it streams.",
                instruct="A warm female narrator with a calm, clear voice.",
                language="English",
            )
        )
        audio = np.concatenate([c for c, _ in chunks])
        assert len(audio) > 24000 * 1.5 and np.isfinite(audio).all()
        assert runtime.runtime_info()["backend"] == "mlx"
        with pytest.raises(ValueError, match="non-empty voice description"):
            list(runtime.stream_voice_design("No description.", instruct="  "))
    finally:
        runtime.unload()


def test_clone_from_a_three_second_reference_on_mlx(custom_runtime, tmp_path):
    if not _cached(BASE):
        pytest.skip(f"{BASE} is not in the local Hugging Face cache")
    import soundfile as sf

    from abstractvoice.qwen3_tts.runtime import Qwen3TTSRuntime

    # A ~3 s reference spoken by the CustomVoice checkpoint, with its transcript (ICL).
    ref_audio, sr = custom_runtime.synthesize_custom_voice(REFERENCE_TEXT, speaker="serena", language="English")
    ref_path = tmp_path / "reference.wav"
    sf.write(ref_path, ref_audio, sr)
    base = Qwen3TTSRuntime(BASE, allow_downloads=False)
    try:
        prompt = base.build_clone_prompt(ref_audio=str(ref_path), ref_text=REFERENCE_TEXT, x_vector_only=False)
        assert prompt.icl_mode and prompt.ref_codes is not None and prompt.ref_codes.shape[1] == 16
        chunks = list(base.stream_clone("The cloned voice speaks a new sentence, in the same voice.", clone_prompt=prompt, language="English"))
        audio = np.concatenate([c for c, _ in chunks])
        assert len(audio) > 24000 * 1.5 and np.isfinite(audio).all()
        assert audio[0] == 0.0 and audio[-1] == 0.0
        xvec = base.build_clone_prompt(ref_audio=str(ref_path), ref_text=None, x_vector_only=True)
        buffered, _ = base.synthesize_clone("Speaker-vector cloning also works.", clone_prompt=xvec, language="English")
        assert len(buffered) > 24000
        with pytest.raises(ValueError, match="ref_text is required"):
            base.build_clone_prompt(ref_audio=str(ref_path), ref_text=None, x_vector_only=False)
    finally:
        base.unload()
