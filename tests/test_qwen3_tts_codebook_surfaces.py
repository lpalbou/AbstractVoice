"""Qwen3-TTS predictor/sampler reach every surface (backlog 0950).

The choice lived only on `Qwen3TTSSettings`; the CLI, REPL, web example and
AbstractCore plugin always used the defaults. Each surface now sets it, and a
typo is rejected before any weights load. No torch needed: nothing here loads a
model.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from abstractvoice.qwen3_tts.runtime import (
    Qwen3TTSSettings,
    normalize_qwen3_tts_predictor,
    normalize_qwen3_tts_sampler,
)


# --- validation ---------------------------------------------------------------


def test_normalizers_default_and_reject():
    assert normalize_qwen3_tts_predictor(None) == "auto"
    assert normalize_qwen3_tts_sampler(" Exponential ") == "exponential"
    with pytest.raises(ValueError, match="one of auto, reference"):
        normalize_qwen3_tts_predictor("fast")
    with pytest.raises(ValueError, match="one of multinomial, exponential"):
        Qwen3TTSSettings(sampler="greedy")


# --- launch flags -------------------------------------------------------------


def test_voice_cli_flags():
    from abstractvoice.examples.voice_cli import parse_args

    args = parse_args(["--qwen3-tts-predictor", "reference", "--qwen3-tts-sampler", "exponential"])
    assert (args.qwen3_tts_predictor, args.qwen3_tts_sampler) == ("reference", "exponential")
    defaults = parse_args([])
    assert (defaults.qwen3_tts_predictor, defaults.qwen3_tts_sampler) == ("auto", "multinomial")
    with pytest.raises(SystemExit):
        parse_args(["--qwen3-tts-sampler", "greedy"])


def test_repl_flags(monkeypatch):
    from abstractvoice.examples import cli_repl

    monkeypatch.setattr(sys, "argv", ["abstractvoice-cli", "--qwen3-tts-sampler", "exponential"])
    args = cli_repl.parse_args()
    assert (args.qwen3_tts_predictor, args.qwen3_tts_sampler) == ("auto", "exponential")
    monkeypatch.setattr(sys, "argv", ["abstractvoice-cli", "--qwen3-tts-predictor", "fast"])
    with pytest.raises(SystemExit):
        cli_repl.parse_args()


def test_repl_passes_the_choice_to_its_voice_manager(monkeypatch):
    from abstractvoice.examples import cli_repl

    built = []

    class _VM:
        def __init__(self, **kwargs):
            built.append(kwargs)

        def __getattr__(self, _name):
            return lambda *a, **k: None

    monkeypatch.setattr(cli_repl, "VoiceManager", _VM)
    monkeypatch.setattr(cli_repl.VoiceREPL, "_init_readline", lambda self: None)
    try:
        cli_repl.VoiceREPL(
            tts_engine="openai",
            qwen3_tts_predictor="reference",
            qwen3_tts_sampler="exponential",
        )
    except Exception:
        pass  # later REPL setup may need a real manager; only construction matters here
    assert built and built[0]["qwen3_tts_predictor"] == "reference"
    assert built[0]["qwen3_tts_sampler"] == "exponential"


def test_web_example_flags():
    from abstractvoice.examples.web_ui import parse_args

    args = parse_args(["--qwen3-tts-predictor", "reference"])
    assert (args.qwen3_tts_predictor, args.qwen3_tts_sampler) == ("reference", "multinomial")
    with pytest.raises(SystemExit):
        parse_args(["--qwen3-tts-predictor", "fast"])


# --- library plumbing ---------------------------------------------------------


def test_voice_manager_forwards_the_choice_to_the_adapter_factory(monkeypatch):
    import abstractvoice.vm.manager as manager

    seen = {}

    def fake_create(**kwargs):
        seen.update(kwargs)
        return None, kwargs["engine"]

    monkeypatch.setattr(manager, "create_tts_adapter", fake_create)
    vm = manager.VoiceManager(tts_engine="qwen3-tts", stt_engine="openai", qwen3_tts_sampler="exponential")
    assert (seen["qwen3_tts_predictor"], seen["qwen3_tts_sampler"]) == ("auto", "exponential")
    assert (vm.qwen3_tts_predictor, vm.qwen3_tts_sampler) == ("auto", "exponential")


def test_voice_manager_rejects_a_typo_before_building_anything(monkeypatch):
    import abstractvoice.vm.manager as manager

    monkeypatch.setattr(manager, "create_tts_adapter", lambda **_k: pytest.fail("must validate first"))
    with pytest.raises(ValueError, match="predictor"):
        manager.VoiceManager(tts_engine="qwen3-tts", qwen3_tts_predictor="fast")


def test_registry_builds_the_adapter_with_the_choice():
    from abstractvoice.adapters.tts_registry import create_tts_adapter

    adapter, resolved = create_tts_adapter(
        engine="qwen3-tts",
        language="en",
        allow_downloads=False,
        auto_load=False,
        qwen3_tts_predictor="reference",
        qwen3_tts_sampler="exponential",
    )
    assert resolved == "qwen3-tts"
    assert (adapter._runtime.settings.predictor, adapter._runtime.settings.sampler) == ("reference", "exponential")


class _FakeRuntime:
    def __init__(self, loaded: bool):
        self.settings = Qwen3TTSSettings()
        self.is_loaded = loaded
        self.unloads = 0
        self.model_id = "fake"

    def unload(self):
        self.unloads += 1
        self.is_loaded = False


def test_adapter_change_unloads_a_resident_model_only_when_it_changes():
    from abstractvoice.adapters.tts_qwen3_tts import Qwen3TTSAdapter

    runtime = _FakeRuntime(loaded=True)
    adapter = Qwen3TTSAdapter(runtime=runtime)
    adapter.set_codebook_generation(predictor="auto", sampler="multinomial")
    assert runtime.unloads == 0
    adapter.set_codebook_generation(predictor="auto", sampler="exponential")
    assert runtime.settings.sampler == "exponential" and runtime.unloads == 1
    with pytest.raises(ValueError):
        adapter.set_codebook_generation(predictor="fast", sampler="exponential")
    assert runtime.settings.predictor == "auto"


def test_cloner_builds_its_engine_with_the_choice_and_drops_it_on_change():
    from abstractvoice.cloning.manager import VoiceCloner

    cloner = VoiceCloner(allow_downloads=False, qwen3_tts_predictor="reference")
    engine = cloner._get_engine("qwen3-tts")
    assert engine._settings.predictor == "reference"
    cloner.set_qwen3_tts_codebook_generation(predictor="reference", sampler="exponential")
    rebuilt = cloner._get_engine("qwen3-tts")
    assert rebuilt is not engine
    assert (rebuilt._settings.predictor, rebuilt._settings.sampler) == ("reference", "exponential")


def test_voice_manager_runtime_change_reaches_adapter_and_cloner():
    from abstractvoice.vm.tts_mixin import TtsMixin

    calls = []
    host = SimpleNamespace(
        qwen3_tts_predictor="auto",
        qwen3_tts_sampler="multinomial",
        tts_adapter=SimpleNamespace(set_codebook_generation=lambda **k: calls.append(("adapter", k))),
        _voice_cloner=SimpleNamespace(set_qwen3_tts_codebook_generation=lambda **k: calls.append(("cloner", k))),
    )
    out = TtsMixin.set_qwen3_tts_codebook_generation(host, sampler="exponential")
    assert out == {"predictor": "auto", "sampler": "exponential"}
    assert calls == [
        ("adapter", {"predictor": "auto", "sampler": "exponential"}),
        ("cloner", {"predictor": "auto", "sampler": "exponential"}),
    ]
    with pytest.raises(ValueError):
        TtsMixin.set_qwen3_tts_codebook_generation(host, predictor="fast")
    assert host.qwen3_tts_predictor == "auto"


# --- AbstractCore plugin settings ---------------------------------------------


class _Owner:
    def __init__(self, config):
        self.config = config


def _recording_voice_manager(monkeypatch):
    import abstractvoice.integrations.abstractcore_plugin as plugin
    import abstractvoice.voice_manager as voice_manager_module

    built = []

    class _VM:
        def __init__(self, **kwargs):
            built.append(kwargs)

    monkeypatch.setattr(voice_manager_module, "VoiceManager", _VM)
    monkeypatch.setattr(plugin, "_VM_CACHE", {})
    return plugin, built


def test_plugin_settings_keys_reach_the_voice_manager(monkeypatch):
    plugin, built = _recording_voice_manager(monkeypatch)
    cap = plugin._VoiceCapability(
        _Owner({"voice_qwen3_tts_predictor": "reference", "voice_qwen3_tts_sampler": "exponential"})
    )
    cap._get_vm()
    assert (built[0]["qwen3_tts_predictor"], built[0]["qwen3_tts_sampler"]) == ("reference", "exponential")


def test_plugin_settings_typo_fails_loudly(monkeypatch):
    plugin, built = _recording_voice_manager(monkeypatch)
    cap = plugin._VoiceCapability(_Owner({"voice_qwen3_tts_sampler": "greedy"}))
    with pytest.raises(ValueError, match="sampler"):
        cap._get_vm()
    assert built == []


# --- web example control ------------------------------------------------------


def test_web_state_reports_and_applies_the_choice():
    from abstractvoice.examples.web_ui import ExampleState

    applied = []
    fake_vm = SimpleNamespace(set_qwen3_tts_codebook_generation=lambda **k: applied.append(k))
    state = ExampleState(
        language="en",
        tts_engine="openai",
        stt_engine="openai",
        whisper_model="base",
        qwen3_tts_sampler="exponential",
        voice_manager_factory=lambda _state: fake_vm,
    )
    assert state.status_dict()["qwen3_tts"] == {"predictor": "auto", "sampler": "exponential"}
    state.get_voice_manager()
    out = state.set_qwen3_tts_codebook(predictor="reference", sampler=None)
    assert out["predictor"] == "reference" and out["sampler"] == "exponential"
    assert applied == [{"predictor": "reference", "sampler": "exponential"}]
    with pytest.raises(ValueError):
        state.set_qwen3_tts_codebook(predictor="fast", sampler=None)
    assert state.qwen3_tts_predictor == "reference"


def test_web_endpoint_rejects_a_typo_with_400():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from abstractvoice.examples.web_ui import create_app

    app = create_app(tts_engine="openai", voice_manager_factory=lambda _state: SimpleNamespace())
    client = TestClient(app)
    ok = client.post("/api/qwen3-tts/codebook", json={"sampler": "exponential"})
    assert ok.status_code == 200 and ok.json()["sampler"] == "exponential"
    bad = client.post("/api/qwen3-tts/codebook", json={"predictor": "fast"})
    assert bad.status_code == 400
    assert client.get("/api/status").json()["qwen3_tts"]["sampler"] == "exponential"
