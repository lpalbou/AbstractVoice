"""Real-weight Qwen family proofs through AbstractCore's production plugin registry.

Run with ABSTRACTVOICE_RUN_QWEN3_TTS_TESTS=1 and pytest -m model_download.
All five checkpoints must be prefetched; inference is strictly offline. Install
AbstractCore alongside AbstractVoice to exercise the real capability boundary.
"""
from __future__ import annotations

import io
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

pytestmark = [
    pytest.mark.model_download,
    pytest.mark.skipif(os.environ.get("ABSTRACTVOICE_RUN_QWEN3_TTS_TESTS") != "1", reason="opt-in real Qwen checkpoints"),
]

CUSTOM = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
LARGE_CUSTOM = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
DESIGN = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
BASE = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
LARGE_BASE = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
REFERENCE_TEXT = "This reference voice belongs to the abstract framework test suite."


def assert_audio(data):
    assert data[:4] == b"RIFF"
    audio, sr = sf.read(io.BytesIO(data))
    assert sr == 24000 and len(audio) > sr * 0.2
    assert np.isfinite(audio).all()
    assert np.sqrt(np.mean(audio ** 2)) > 1e-5
    assert audio[0] == 0.0 and audio[-1] == 0.0  # de-clicked utterance edges


@pytest.fixture
def core_voice(tmp_path, monkeypatch):
    from abstractvoice.cloning.store import VoiceCloneStore
    import abstractvoice.integrations.abstractcore_plugin as plugin

    registry = pytest.importorskip("abstractcore.capabilities.registry")
    store = VoiceCloneStore(tmp_path / "voices")
    monkeypatch.setattr("abstractvoice.cloning.manager.VoiceCloneStore", lambda: store)
    monkeypatch.setattr("abstractvoice.cloning.store.VoiceCloneStore", lambda: store)
    monkeypatch.setattr(plugin, "_VM_CACHE", {})
    owner = SimpleNamespace(config={
        "voice_tts_engine": "qwen3-tts", "voice_allow_downloads": False,
        "voice_cloning_engine": "qwen3-tts", "voice_language": "en",
    })
    core = registry.CapabilityRegistry(owner, preferred_backends={"voice": "abstractvoice:default"})
    # This resolves the installed entry point, not a fake backend.
    yield core.voice
    for vm in plugin._VM_CACHE.values():
        vm.unload_tts_engine()
        cloner = getattr(vm, "_voice_cloner", None)
        if cloner is not None:
            cloner.unload_all_engines()


@pytest.fixture(scope="module")
def reference_audio():
    from abstractvoice.adapters.tts_qwen3_tts import Qwen3TTSAdapter

    adapter = Qwen3TTSAdapter(model_id=CUSTOM, allow_downloads=False)
    try:
        data = adapter.synthesize_to_bytes(REFERENCE_TEXT)
        assert_audio(data)
        return data
    finally:
        adapter.unload()


@pytest.mark.parametrize("model", [CUSTOM, LARGE_CUSTOM])
def test_custom_voice_through_core_registry_buffered_and_streamed(core_voice, model):
    profiles = core_voice.voice_catalog(provider="qwen3-tts", model=model)["profiles"]
    assert len(profiles) == 9
    kwargs = {"provider": "qwen3-tts", "model": model, "voice": "aiden"}
    if model == LARGE_CUSTOM:
        kwargs["instructions"] = "Speak warmly and clearly."
    assert_audio(core_voice.tts("The voice integration is ready.", **kwargs))
    events = list(core_voice.tts_stream("Streaming speech is working.", **kwargs))
    audio_events = [e for e in events if e["type"] == "audio"]
    assert audio_events and events[-1]["type"] == "done"
    for event in audio_events:
        assert_audio(event["audio"])
        assert event["model"] == model


def test_voice_design_through_core_registry(core_voice):
    kwargs = {"provider": "qwen3-tts", "model": DESIGN, "instructions": "A warm female narrator with a calm, clear voice."}
    assert_audio(core_voice.tts("This voice was described in words.", **kwargs))
    events = list(core_voice.tts_stream("The designed voice can stream too.", **kwargs))
    audio_events = [e for e in events if e["type"] == "audio"]
    assert audio_events and events[-1]["type"] == "done"
    for event in audio_events:
        assert_audio(event["audio"])


def test_voice_design_cli_file(tmp_path):
    output = tmp_path / "design.wav"
    result = subprocess.run([
        sys.executable, "-m", "abstractvoice.examples.voice_cli",
        "--tts-engine", "qwen3-tts", "--tts-model", DESIGN,
        "--instructions", "A warm, clear narrator.",
        "--prompt", "The command line can design a voice.", "--output", str(output),
    ], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert_audio(output.read_bytes())


@pytest.mark.parametrize("model", [BASE, LARGE_BASE])
def test_clone_checkpoint_through_core_registry(core_voice, reference_audio, model):
    voice_id = core_voice.clone(reference_audio, provider="qwen3-tts", model=model, reference_text=REFERENCE_TEXT)
    assert isinstance(voice_id, str) and voice_id
    catalog = core_voice.voice_catalog(provider="qwen3-tts", model=model)
    assert any(v["voice_id"] == voice_id for v in catalog["cloned_voices"])
    kwargs = {"provider": "qwen3-tts", "model": model, "voice": voice_id}
    assert_audio(core_voice.tts("The cloned voice is speaking through the plugin.", **kwargs))
    events = list(core_voice.tts_stream("The cloned voice can stream too.", **kwargs))
    assert any(e["type"] == "audio" for e in events)
    for event in events:
        if event["type"] == "audio":
            assert_audio(event["audio"])
