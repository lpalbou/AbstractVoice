"""Regression coverage for Qwen checkpoint identity and public entry points."""
from __future__ import annotations

import io
import json
from importlib.util import find_spec
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from abstractvoice.adapters.tts_qwen3_tts import Qwen3TTSAdapter
from abstractvoice.cloning.manager import VoiceCloner
from abstractvoice.cloning.store import VoiceCloneStore
from abstractvoice.integrations.abstractcore_plugin import _VoiceCapability
from abstractvoice.qwen3_tts.runtime import Qwen3TTSSettings

CUSTOM = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
LARGE_CUSTOM = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
BASE = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
LARGE_BASE = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
DESIGN = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"

# VoiceManager(tts_engine="qwen3-tts") builds the real adapter, which needs the
# qwen3-tts extra's runtimes importable (never loaded here). CI installs `.[test]`.
requires_qwen_runtime = pytest.mark.skipif(
    find_spec("torch") is None or find_spec("transformers") is None,
    reason="needs the qwen3-tts extra (torch + transformers)",
)


class Runtime:
    def __init__(self, model_id=CUSTOM):
        self.model_id = model_id
        self.is_loaded = True
        self.settings = Qwen3TTSSettings()
        self.calls = []

    def model_type(self):
        if not self.is_loaded:
            raise RuntimeError("not cached")
        return "voice_design" if self.model_id == DESIGN else "custom_voice"

    def speaker_names(self):
        return ["aiden", "serena"] if self.is_loaded else []

    def _ensure_loaded(self):
        self.is_loaded = True

    def synthesize_custom_voice(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return np.full(240, 0.1, dtype=np.float32), 24000

    def synthesize_voice_design(self, text, **kwargs):
        return self.synthesize_custom_voice(text, **kwargs)


def wav_bytes():
    buf = io.BytesIO()
    sf.write(buf, np.zeros(2400), 24000, format="WAV")
    return buf.getvalue()


@pytest.mark.parametrize("model", [CUSTOM, DESIGN])
def test_buffered_long_text_preserves_all_segments_and_selectors(model):
    runtime = Runtime(model)
    adapter = Qwen3TTSAdapter(runtime=runtime)
    text = "A complete sentence to be spoken. " * 60 + "The very last sentence."
    data = adapter.synthesize_to_bytes_with_voice(text, voice="serena", instructions="Calm narrator")
    audio, sr = sf.read(io.BytesIO(data))
    assert len(runtime.calls) > 1
    assert all(len(t) <= adapter.get_max_chars() for t, _ in runtime.calls)
    assert " ".join(t for t, _ in runtime.calls) == text.strip()
    assert all(k["instruct"] == "Calm narrator" for _, k in runtime.calls)
    assert len(audio) == len(runtime.calls) * 240 and sr == 24000


def test_lazy_synthesis_loads_before_choosing_default_speaker():
    runtime = Runtime()
    runtime.is_loaded = False
    adapter = Qwen3TTSAdapter(runtime=runtime, auto_load=False)
    assert adapter.synthesize_to_bytes("Hello")[:4] == b"RIFF"
    assert runtime.calls[0][1]["speaker"] == "aiden"


@pytest.fixture
def two_checkpoints(tmp_path, monkeypatch):
    pytest.importorskip("huggingface_hub")
    hub = tmp_path / "hub"
    for model in (CUSTOM, LARGE_CUSTOM):
        snap = hub / ("models--" + model.replace("/", "--")) / "snapshots" / "revision"
        snap.mkdir(parents=True)
        (snap / "model.safetensors").touch()
        (snap / "config.json").write_text(json.dumps({
            "tts_model_type": "custom_voice", "talker_config": {"spk_id": {"aiden": 0, "serena": 1}},
        }))
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(hub))
    store = VoiceCloneStore(tmp_path / "voices")
    monkeypatch.setattr("abstractvoice.cloning.store.VoiceCloneStore", lambda: store)
    return hub


@requires_qwen_runtime
def test_plugin_discovery_keeps_shared_speakers_for_each_model(two_checkpoints):
    cap = _VoiceCapability(SimpleNamespace(config={}))
    for model in (CUSTOM, LARGE_CUSTOM):
        voices = cap.list_tts_voices(provider="qwen3-tts", model=model)
        assert {v["profile_id"] for v in voices} == {"aiden", "serena"}
        assert all(v["params"]["model"] == model for v in voices)


@pytest.mark.parametrize("source", ["bytes", "path"])
def test_clone_model_survives_restart_and_switches_without_reusing_wrong_weights(tmp_path, source):
    store = VoiceCloneStore(tmp_path / "voices")
    cloner = VoiceCloner(store=store, default_engine="qwen3-tts", default_model=LARGE_BASE)
    if source == "path":
        ref = tmp_path / "ref.wav"
        ref.write_bytes(wav_bytes())
        voice_id = cloner.clone_voice(str(ref), reference_text="Reference")
    else:
        voice_id = cloner.clone_voice_from_wav_bytes(wav_bytes(), reference_text="Reference")
    assert store.get_voice(voice_id).meta["model_id"] == LARGE_BASE
    restarted = VoiceCloner(store=store, default_engine="qwen3-tts", allow_downloads=False)
    large = restarted._get_engine_for_voice(store.get_voice(voice_id))
    assert large._model_id == LARGE_BASE
    # A legacy record without a model always means the original 0.6B model,
    # irrespective of the new process's configured default.
    legacy = store.create_voice_from_wav_bytes(wav_bytes(), engine="qwen3-tts", reference_text="Reference")
    small = restarted._get_engine_for_voice(store.get_voice(legacy))
    assert small._model_id == BASE and small is not large
    again = restarted._get_engine_for_voice(store.get_voice(voice_id))
    assert again._model_id == LARGE_BASE and again is not small
    assert list(restarted._engines) == ["qwen3-tts"]


def test_clone_rejects_wrong_checkpoint_role_before_persistence(tmp_path):
    store = VoiceCloneStore(tmp_path)
    cloner = VoiceCloner(store=store, default_engine="qwen3-tts")
    with pytest.raises(ValueError, match="Base checkpoint"):
        cloner.clone_voice_from_wav_bytes(wav_bytes(), model=CUSTOM)
    assert store.list_voices() == []


@requires_qwen_runtime
def test_plugin_clone_routes_requested_checkpoint_and_persists_it(tmp_path, monkeypatch):
    import abstractvoice.integrations.abstractcore_plugin as plugin
    plugin._VM_CACHE.clear()
    store = VoiceCloneStore(tmp_path)
    monkeypatch.setattr("abstractvoice.cloning.manager.VoiceCloneStore", lambda: store)
    monkeypatch.setattr("abstractvoice.cloning.store.VoiceCloneStore", lambda: store)
    # Hermetic cache state: the default CustomVoice checkpoint is on disk (as on
    # any machine that speaks with Qwen), so a provider-filtered listing is
    # answered from disk. Never depend on the developer's real HF cache.
    monkeypatch.setattr("abstractvoice.local_models.hf_repo_is_cached", lambda model_id: model_id == CUSTOM)
    owner = SimpleNamespace(config={"voice_tts_engine": "qwen3-tts", "voice_allow_downloads": False})
    cap = _VoiceCapability(owner)
    voice_id = cap.clone(wav_bytes(), provider="qwen3-tts", model=LARGE_BASE, reference_text="Reference")
    voice = store.get_voice(voice_id)
    assert voice.engine == "qwen3-tts" and voice.meta["model_id"] == LARGE_BASE
    vms = list(plugin._VM_CACHE.values())
    assert any(vm.cloning_model == LARGE_BASE for vm in vms)
    assert all(not vm.tts_adapter.is_engine_loaded() for vm in vms)
    catalog = cap.voice_catalog(provider="qwen3-tts", model=LARGE_BASE)
    assert any(v["voice_id"] == voice_id for v in catalog["cloned_voices"])
    assert not cap.voice_catalog(provider="qwen3-tts", model=BASE)["cloned_voices"]
    legacy = store.create_voice_from_wav_bytes(wav_bytes(), engine="qwen3-tts", reference_text="Reference")
    assert {v["voice_id"] for v in cap.voice_catalog(provider="qwen3-tts", model=BASE)["cloned_voices"]} == {legacy}
    assert {v["voice_id"] for v in cap.voice_catalog(provider="qwen3-tts", model=LARGE_BASE)["cloned_voices"]} == {voice_id}


@pytest.mark.parametrize("config_model", [None, LARGE_BASE])
def test_plugin_cloning_model_is_a_setting_not_an_env_var(monkeypatch, config_model):
    import abstractvoice.integrations.abstractcore_plugin as plugin
    plugin._VM_CACHE.clear()
    monkeypatch.setenv("ABSTRACTVOICE_CLONING_MODEL", "Qwen/from-the-environment")
    config = {"voice_tts_engine": "qwen3-tts", "voice_allow_downloads": False}
    if config_model:
        config["voice_cloning_model"] = config_model
    vm = _VoiceCapability(SimpleNamespace(config=config))._get_vm()
    assert vm.cloning_model == config_model


@requires_qwen_runtime
def test_plugin_streams_design_instructions_without_leaking_state():
    from abstractvoice import VoiceManager

    runtime = Runtime(DESIGN)
    vm = VoiceManager(tts_engine="qwen3-tts", tts_model=DESIGN, allow_downloads=False)
    vm.tts_adapter = Qwen3TTSAdapter(runtime=runtime)
    vm.tts_adapter.set_instructions("Session description")
    cap = _VoiceCapability(SimpleNamespace(config={"voice_manager_instance": vm}))
    events = list(cap.tts_stream("Hello from a designed voice.", instructions="Per-call description"))
    assert any(e["type"] == "audio" and e["audio"][:4] == b"RIFF" for e in events)
    assert runtime.calls[-1][1]["instruct"] == "Per-call description"
    assert vm.tts_adapter._instructions == "Session description"
    assert events[-1]["type"] == "done"


@requires_qwen_runtime
def test_stream_rejects_instructions_on_unsupported_small_model():
    from abstractvoice import VoiceManager

    vm = VoiceManager(tts_engine="qwen3-tts", allow_downloads=False)
    vm.tts_adapter = Qwen3TTSAdapter(runtime=Runtime())
    with pytest.raises(ValueError, match="does not support instructions"):
        list(vm.speak_to_audio_chunks("Hello", instructions="Calm"))


def test_cli_and_repl_accept_qwen_and_download_exact_repo_case(monkeypatch, capsys):
    from abstractvoice.examples import voice_cli, web_ui, cli_repl

    argv = ["--cloning-engine", "qwen3-tts", "--cloning-model", LARGE_BASE]
    assert voice_cli.parse_args(argv).cloning_model == LARGE_BASE
    assert web_ui.parse_args(argv).cloning_model == LARGE_BASE
    monkeypatch.setattr("sys.argv", ["abstractvoice", *argv])
    assert cli_repl.parse_args().cloning_model == LARGE_BASE
    calls = []
    repl = cli_repl.VoiceREPL.__new__(cli_repl.VoiceREPL)
    repl.voice_manager = object()
    monkeypatch.setattr("abstractvoice.qwen3_tts.runtime.prefetch_qwen3_tts", lambda **kw: calls.append(kw))
    repl.do_tts_download(f"qwen3-tts {LARGE_CUSTOM}")
    repl.do_cloning_download(f"qwen3-tts {LARGE_BASE}")
    assert calls == [{"model_id": LARGE_CUSTOM}, {"model_id": LARGE_BASE}]


def test_web_qwen_model_switch_and_instructions_are_forwarded():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from abstractvoice.examples.web_ui import create_app

    class VM:
        tts_model = None
        calls = []
        def set_tts_engine(self, engine, *, tts_model=None):
            self.calls.append((engine, tts_model))
            return engine
        def speak_to_bytes(self, text, **kw):
            self.calls.append((text, kw))
            return wav_bytes()

    vm = VM()
    client = TestClient(create_app(voice_manager_factory=lambda _: vm))
    page = client.get('/').text
    assert page.count('value="qwen3-tts"') == 2
    assert 'id="tts-model"' in page and 'id="tts-instructions"' in page
    response = client.post('/api/tts/provider', json={"provider": "qwen3-tts", "model": DESIGN})
    assert response.status_code == 200
    assert vm.calls[-1] == ("qwen3-tts", DESIGN)
    response = client.post('/api/tts', json={"input": "Hello", "instructions": "Calm narrator"})
    assert response.status_code == 200
    assert vm.calls[-1][1]["instructions"] == "Calm narrator"
    response = client.post('/api/tts/provider', json={"provider": "piper"})
    assert response.status_code == 200
    assert vm.calls[-1] == ("piper", "")  # no foreign checkpoint leaks into Piper


@pytest.mark.parametrize("old_provider,old_model,new_provider", [
    ("qwen3-tts", LARGE_CUSTOM, "piper"),
    ("openai", "gpt-4o-mini-tts", "qwen3-tts"),
    ("qwen3-tts", LARGE_CUSTOM, "qwen3-tts"),
])
def test_manager_engine_switch_only_preserves_same_provider_model(monkeypatch, old_provider, old_model, new_provider):
    from abstractvoice import VoiceManager

    vm = VoiceManager(tts_engine="qwen3-tts", allow_downloads=False)
    vm.tts_adapter = Qwen3TTSAdapter(runtime=Runtime())
    vm.tts_adapter.engine_id = old_provider
    vm.tts_model = old_model
    calls = []

    def factory(**kwargs):
        calls.append(kwargs)
        adapter = Qwen3TTSAdapter(runtime=Runtime(kwargs["model_id"] or CUSTOM))
        adapter.engine_id = kwargs["engine"]
        return adapter, kwargs["engine"]

    monkeypatch.setattr("abstractvoice.vm.tts_mixin.create_tts_adapter", factory)
    assert vm.set_tts_engine(new_provider, auto_load=False) == new_provider
    expected = old_model if old_provider == new_provider else None
    assert calls[-1]["model_id"] == expected
    assert vm.tts_model == expected


@pytest.mark.parametrize("initial_engine,model,next_engine", [
    ("qwen3-tts", LARGE_CUSTOM, "piper"),
    ("openai", "gpt-4o-mini-tts", "qwen3-tts"),
])
def test_repl_does_not_forward_startup_model_to_another_provider(initial_engine, model, next_engine):
    from abstractvoice.examples.cli_repl import VoiceREPL

    calls = []
    vm = SimpleNamespace(set_tts_engine=lambda engine, **kw: calls.append((engine, kw)) or engine, tts_model=None)
    repl = VoiceREPL.__new__(VoiceREPL)
    repl.voice_manager = vm
    repl.current_language = "en"
    repl._initial_tts_engine = initial_engine
    repl._initial_tts_model = model
    repl._warm_tts_audio_output_async = lambda: None
    repl.do_tts_engine(next_engine)
    assert calls == [(next_engine, {})]
    assert repl._initial_tts_model is None


def test_repl_microphone_clone_prefers_qwen_and_accepts_all_its_languages(monkeypatch):
    from abstractvoice.examples.cli_repl import VoiceREPL, parse_args

    repl = VoiceREPL.__new__(VoiceREPL)
    repl.voice_manager = SimpleNamespace(_tts_engine_name="qwen3-tts")
    assert repl._default_mic_clone_engine() == "qwen3-tts"
    for language in ("ja", "ko", "it", "pt"):
        monkeypatch.setattr("sys.argv", ["abstractvoice", "--language", language, "--tts-engine", "qwen3-tts"])
        assert parse_args().language == language


def test_module_launcher_passes_qwen_languages_to_repl(monkeypatch):
    import sys
    from abstractvoice.__main__ import main

    calls = []
    monkeypatch.setattr("abstractvoice.examples.cli_repl.main", lambda: calls.append(sys.argv[1:]))
    monkeypatch.setattr(sys, "argv", ["abstractvoice", "cli", "--language", "ja", "--tts-engine", "qwen3-tts"])
    main()
    assert calls == [["--language", "ja", "--tts-engine", "qwen3-tts"]]


@requires_qwen_runtime
@pytest.mark.parametrize("route", ["/api/voices/clone", "/v1/voice/clone"])
def test_web_clone_model_is_persisted_through_real_manager(tmp_path, route):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from abstractvoice import VoiceManager
    from abstractvoice.examples.web_ui import create_app

    vm = VoiceManager(tts_engine="qwen3-tts", cloning_engine="qwen3-tts", allow_downloads=False)
    vm._voice_cloner = VoiceCloner(store=VoiceCloneStore(tmp_path), default_engine="qwen3-tts", allow_downloads=False)
    client = TestClient(create_app(voice_manager_factory=lambda _: vm))
    response = client.post(route, files={"file": ("ref.wav", wav_bytes(), "audio/wav")}, data={
        "provider": "qwen3-tts", "model": LARGE_BASE, "reference_text": "Reference", "validate": "false",
    })
    assert response.status_code == 200, response.text
    voice_id = response.json()["voice_id"]
    assert vm.get_cloned_voice(voice_id)["meta"]["model_id"] == LARGE_BASE
    assert not vm.tts_adapter.is_engine_loaded()


def test_qwen_extra_requires_torch_compatible_with_transformers_five():
    from pathlib import Path
    from packaging.requirements import Requirement

    tomllib = pytest.importorskip("tomllib")
    project = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    requirements = [Requirement(item) for item in project["project"]["optional-dependencies"]["qwen3-tts"]]
    torch = next(item for item in requirements if item.name == "torch")
    assert "2.4.0" in torch.specifier and "2.3.1" not in torch.specifier
