"""Real-weights check of voice design through the AbstractCore plugin.

A voice is designed from a description, listed with the cloned voices, spoken, and
deleted. Needs the qwen3-tts extra plus Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign and
Qwen/Qwen3-TTS-12Hz-1.7B-Base in the Hugging Face cache. Marked ``model_download``;
enable with ``ABSTRACTVOICE_RUN_VOICE_DESIGN_TESTS=1``.
"""
from __future__ import annotations

import io
import os

import pytest

pytestmark = [
    pytest.mark.model_download,
    pytest.mark.skipif(
        os.environ.get("ABSTRACTVOICE_RUN_VOICE_DESIGN_TESTS", "0") != "1",
        reason="set ABSTRACTVOICE_RUN_VOICE_DESIGN_TESTS=1 to run (needs the 1.7B VoiceDesign and Base checkpoints)",
    ),
]


def test_designed_voice_round_trip_through_the_plugin():
    sf = pytest.importorskip("soundfile")
    from abstractvoice.integrations.abstractcore_plugin import _VoiceCapability

    class _Owner:
        config = {"voice_allow_downloads": False}

    cap = _VoiceCapability(_Owner())
    created = cap.design_voice(
        "A calm, warm female voice in her thirties, clear and friendly, speaking at a relaxed pace.",
        name="Design check",
        language="en",
    )
    voice_id = created["voice_id"]
    try:
        assert created["ok"] is True and created["engine"] == "qwen3-tts"
        assert created["meta"]["designed"] is True
        listed = {v["voice_id"]: v for v in cap.voice_catalog(provider="qwen3-tts")["cloned_voices"]}
        assert listed[voice_id]["tags"]["designed"] is True
        out = cap.tts("Good morning, this voice was designed from a description.", voice=voice_id, one_shot=True)
        audio, sr = sf.read(io.BytesIO(out["audio_bytes"] if isinstance(out, dict) else out))
        assert sr == 24000 and 1.0 < len(audio) / sr < 15.0
        assert float((audio**2).mean()) ** 0.5 > 0.01
    finally:
        assert cap.delete_cloned_voice(voice_id)["deleted"] is True
