"""Qwen3-ASR end-to-end proof on the real checkpoint (backlog 0946).

Gated: set ``ABSTRACTVOICE_RUN_QWEN3_ASR_TESTS=1`` and prefetch the snapshot
(``abstractvoice-prefetch --stt-hf Qwen/Qwen3-ASR-0.6B``, ~1.9 GB). The
reference clip is spoken by macOS ``say``, so the test skips elsewhere. Marked
``model_download`` so CI's default run skips it.
"""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest

pytestmark = [
    pytest.mark.model_download,
    pytest.mark.skipif(
        os.environ.get("ABSTRACTVOICE_RUN_QWEN3_ASR_TESTS", "0") != "1",
        reason="set ABSTRACTVOICE_RUN_QWEN3_ASR_TESTS=1 to run the Qwen3-ASR end-to-end test",
    ),
    pytest.mark.skipif(
        shutil.which("say") is None or shutil.which("afconvert") is None,
        reason="the reference clip is spoken by macOS say",
    ),
]

MODEL = "Qwen/Qwen3-ASR-0.6B"
SENTENCE = "The quick brown fox jumps over the lazy dog. Speech recognition works on this computer."


def _words(text: str) -> list[str]:
    return "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in text).split()


def test_real_checkpoint_transcribes_reference_clip(tmp_path):
    from abstractvoice.adapters.stt_transformers_asr import TransformersASRAdapter

    aiff, wav = tmp_path / "ref.aiff", tmp_path / "ref.wav"
    subprocess.run(["say", "-v", "Samantha", "-o", str(aiff), SENTENCE], check=True)
    subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(aiff), str(wav)], check=True)

    adapter = TransformersASRAdapter(model_id=MODEL, device="cpu", allow_downloads=False)
    text = adapter.transcribe(str(wav), language="en")
    assert _words(text) == _words(SENTENCE)
