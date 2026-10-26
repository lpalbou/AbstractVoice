"""Real-weights check of the ``transformers-asr`` Whisper path.

Downloads ``openai/whisper-tiny`` (Apache-2.0, about 150 MB) once, then transcribes
a clip spoken by macOS ``say`` with downloads disabled, so the offline path is the
one exercised. Marked ``model_download``, so CI's default run skips it; enable with
``ABSTRACTVOICE_RUN_WHISPER_HF_TESTS=1``.
"""
from __future__ import annotations

import os
import shutil
import subprocess

import pytest

pytestmark = [
    pytest.mark.model_download,
    pytest.mark.skipif(
        os.environ.get("ABSTRACTVOICE_RUN_WHISPER_HF_TESTS", "0") != "1",
        reason="set ABSTRACTVOICE_RUN_WHISPER_HF_TESTS=1 to run (downloads openai/whisper-tiny)",
    ),
    pytest.mark.skipif(
        shutil.which("say") is None or shutil.which("afconvert") is None,
        reason="the reference clip is spoken by macOS say",
    ),
]

MODEL = "openai/whisper-tiny"
SENTENCE = "The quick brown fox jumps over the lazy dog."


def test_whisper_tiny_transcribes_offline_through_transformers_asr(tmp_path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from abstractvoice.adapters.stt_transformers_asr import TransformersASRAdapter

    aiff, wav = tmp_path / "ref.aiff", tmp_path / "ref.wav"
    subprocess.run(["say", "-v", "Samantha", "-o", str(aiff), SENTENCE], check=True)
    subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(aiff), str(wav)], check=True)

    # Populate the cache, then exercise the offline path from it.
    TransformersASRAdapter(model_id=MODEL, allow_downloads=True)._ensure_loaded()
    adapter = TransformersASRAdapter(model_id=MODEL, allow_downloads=False)

    text = adapter.transcribe(str(wav), language="en").lower()

    assert "fox" in text and "dog" in text, text
    assert adapter.runtime_info()["resolved_device"]
