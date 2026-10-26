"""Real-weights check of the built-in voice pack: a stored voice speaks, on the default backend.

Needs the qwen3-tts extra and Qwen/Qwen3-TTS-12Hz-1.7B-Base in the Hugging Face cache.
Marked ``model_download``; enable with ``ABSTRACTVOICE_RUN_VOICE_PACK_TESTS=1``.
"""
from __future__ import annotations

import io
import os

import pytest

pytestmark = [
    pytest.mark.model_download,
    pytest.mark.skipif(
        os.environ.get("ABSTRACTVOICE_RUN_VOICE_PACK_TESTS", "0") != "1",
        reason="set ABSTRACTVOICE_RUN_VOICE_PACK_TESTS=1 to run (needs the 1.7B Base checkpoint)",
    ),
]


@pytest.mark.parametrize("voice_id", ["fr-female-camille", "en-gb-male-alistair"])
def test_builtin_voice_speaks_one_shot(voice_id, tmp_path, monkeypatch):
    pytest.importorskip("soundfile")
    import soundfile as sf

    from abstractvoice import VoiceManager

    vm = VoiceManager(tts_engine="none", stt_engine="none", allow_downloads=False, cloning_engine="qwen3-tts")
    assert any(v["voice_id"] == voice_id for v in vm.list_cloned_voices())
    audio, sr = sf.read(io.BytesIO(vm.speak_to_bytes("Hello, this is a quick check.", voice=voice_id, one_shot=True)))
    assert sr == 24000 and 1.0 < len(audio) / sr < 10.0
    assert float((audio**2).mean()) ** 0.5 > 0.01
