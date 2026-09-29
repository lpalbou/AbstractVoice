"""`abstractvoice.engine_runtime` and the `unavailable_*` fields it feeds.

A voice listing used to answer "the Supertonic runtime is not installed" with an
empty list and no error. These tests pin the public runtime probe and the
explicit reasons the AbstractCore plugin now publishes instead.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys

import pytest

import abstractvoice.engine_runtime as engine_runtime
import abstractvoice.integrations.abstractcore_plugin as plugin
from abstractvoice.engine_runtime import (
    engine_runtime_installed,
    engine_runtime_status,
    known_engines,
    normalize_engine_id,
)


_REAL_FIND_SPEC = importlib.util.find_spec


def _hide_modules(monkeypatch, *hidden: str, present: tuple[str, ...] = ()) -> None:
    """Make `find_spec` report `hidden` modules as absent and `present` ones as there."""

    def fake_find_spec(name, *args, **kwargs):
        root = name.split(".", 1)[0]
        if root in hidden:
            return None
        if root in present:
            return object()
        return _REAL_FIND_SPEC(name, *args, **kwargs)

    monkeypatch.setattr(engine_runtime.importlib.util, "find_spec", fake_find_spec)


def _clear_voice_env(monkeypatch) -> None:
    for key in (
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "ABSTRACTVOICE_TTS_ENGINE",
        "ABSTRACTVOICE_STT_ENGINE",
        "ABSTRACTVOICE_TTS_MODEL",
        "ABSTRACTVOICE_CLONING_ENGINE",
        "ABSTRACTGATEWAY_VOICE_TTS_ENGINE",
        "ABSTRACTGATEWAY_VOICE_STT_ENGINE",
    ):
        monkeypatch.delenv(key, raising=False)


class _Owner:
    def __init__(self, config=None):
        self.config = dict(config or {})


# --- the public probe ---------------------------------------------------------


def test_missing_runtime_is_reported_with_module_extra_and_command(monkeypatch):
    _hide_modules(monkeypatch, "onnxruntime")
    status = engine_runtime_status("supertonic")
    assert status.installed is False
    assert status.missing_modules == ("onnxruntime",)
    assert status.extra == "supertonic"
    assert status.install_command == 'pip install "abstractvoice[supertonic]"'
    assert "onnxruntime" in status.reason and status.install_command in status.reason
    assert status.to_dict()["installed"] is False
    assert engine_runtime_installed("supertonic") is False


def test_present_runtime_has_no_reason(monkeypatch):
    _hide_modules(monkeypatch, present=("faster_whisper",))
    status = engine_runtime_status("faster_whisper", kind="stt")
    assert status.engine == "faster-whisper"
    assert status.installed is True
    assert status.missing_modules == ()
    assert status.reason is None


def test_alternative_modules_satisfy_one_requirement(monkeypatch):
    _hide_modules(monkeypatch, "piper", present=("piper_phonemize",))
    assert engine_runtime_installed("piper") is True
    _hide_modules(monkeypatch, "piper", "piper_phonemize")
    status = engine_runtime_status("piper")
    assert status.installed is False
    assert status.missing_modules == ("piper or piper_phonemize",)


def test_every_missing_requirement_is_named(monkeypatch):
    _hide_modules(monkeypatch, "torch", "transformers", present=("soundfile",))
    status = engine_runtime_status("transformers-asr")
    assert status.missing_modules == ("torch", "transformers")
    assert "packages torch, transformers are missing" in status.reason


def test_remote_engines_need_no_runtime():
    status = engine_runtime_status("openai")
    assert status.remote is True and status.installed is True and status.install_command is None


def test_unknown_engine_and_wrong_kind_fail_loudly():
    with pytest.raises(ValueError, match="unknown AbstractVoice engine"):
        engine_runtime_status("not-an-engine")
    with pytest.raises(ValueError, match="does not serve"):
        engine_runtime_status("faster-whisper", kind="tts")
    with pytest.raises(ValueError, match="kind must be"):
        known_engines("video")


def test_known_engines_and_aliases():
    assert "supertonic" in known_engines("tts")
    assert "faster-whisper" in known_engines("stt") and "faster-whisper" not in known_engines("tts")
    assert normalize_engine_id("F5-TTS") == "f5_tts"
    assert normalize_engine_id("qwen3_tts") == "qwen3-tts"


def test_probe_imports_no_engine_or_ml_framework():
    code = (
        "import sys\n"
        "from abstractvoice.engine_runtime import engine_runtime_status, known_engines\n"
        "[engine_runtime_status(e) for e in known_engines()]\n"
        "heavy = [m for m in ('torch', 'transformers', 'onnxruntime', 'faster_whisper', 'piper', 'omnivoice', 'f5_tts') if m in sys.modules]\n"
        "print(','.join(heavy))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == ""


def test_plugin_uses_the_public_probe(monkeypatch):
    """The plugin has no runtime table of its own: it asks `engine_runtime_status`."""
    asked = []
    real = engine_runtime.engine_runtime_status

    def spy(engine, **kwargs):
        asked.append(engine)
        return real(engine, **kwargs)

    monkeypatch.setattr(engine_runtime, "engine_runtime_status", spy)
    _hide_modules(monkeypatch, "onnxruntime")
    assert plugin._engine_runtime_available("supertonic") is False
    assert plugin._runtime_installed("stt", "faster_whisper") is not None
    assert asked == ["supertonic", "faster-whisper"]


# --- listings say why they are empty ------------------------------------------


def test_voice_catalog_for_uninstalled_provider_says_why(monkeypatch):
    _clear_voice_env(monkeypatch)
    _hide_modules(monkeypatch, "onnxruntime")
    cap = plugin._VoiceCapability(_Owner())
    catalog = cap.voice_catalog(provider="supertonic", model="supertonic-3")
    assert "supertonic" not in catalog["tts_providers"]
    assert "onnxruntime" in catalog["unavailable_reason"]
    record = catalog["unavailable_providers"]["tts"]["supertonic"]
    assert record["code"] == "runtime_missing"
    assert record["runtime"]["install_command"] == 'pip install "abstractvoice[supertonic]"'
    # The sentence an AbstractCore user reads names AbstractCore's settings, never the
    # standalone extra (operator ruling 2026-09-29).
    for sentence in (record["reason"], catalog["unavailable_reason"]):
        assert 'abstractcore[apple]' in sentence and 'abstractcore[gpu]' in sentence
        assert "abstractvoice[" not in sentence


def test_voice_catalog_distinguishes_missing_model_from_missing_runtime(monkeypatch):
    _clear_voice_env(monkeypatch)
    _hide_modules(monkeypatch, present=("onnxruntime",))
    monkeypatch.setattr(plugin, "_selectable_tts_model_ids_for_provider", lambda *_a, **_k: [])
    cap = plugin._VoiceCapability(_Owner())
    catalog = cap.voice_catalog(provider="supertonic")
    assert catalog["unavailable_providers"]["tts"]["supertonic"]["code"] == "model_not_downloaded"
    assert "no Supertonic model is downloaded" in catalog["unavailable_reason"]


def test_available_provider_has_no_reason(monkeypatch):
    _clear_voice_env(monkeypatch)
    _hide_modules(monkeypatch, present=("onnxruntime",))
    monkeypatch.setattr(
        plugin,
        "_selectable_tts_model_ids_for_provider",
        lambda provider, *_a, **_k: ["supertonic-3"] if plugin._norm_engine_id(provider) == "supertonic" else [],
    )
    cap = plugin._VoiceCapability(_Owner())
    catalog = cap.voice_catalog(provider="supertonic")
    assert "supertonic" in catalog["tts_providers"]
    assert catalog["unavailable_reason"] is None
    assert "supertonic" not in catalog["unavailable_providers"]["tts"]


def test_unfiltered_empty_listing_leads_with_configured_provider(monkeypatch):
    _clear_voice_env(monkeypatch)
    _hide_modules(monkeypatch, "onnxruntime", "piper", "piper_phonemize", "torch", "omnivoice")
    cap = plugin._VoiceCapability(_Owner({"voice_tts_engine": "supertonic"}))
    catalog = cap.voice_catalog(providers_only=True)
    assert catalog["tts_providers"] == []
    assert catalog["unavailable_reason"].startswith("no text-to-speech provider is available: Supertonic is not installed")
    assert catalog["unavailable_providers"]["tts"]["openai"]["code"] == "not_configured"


def test_available_providers_explains_missing_stt_and_cloning(monkeypatch):
    _clear_voice_env(monkeypatch)
    _hide_modules(monkeypatch, "faster_whisper", "omnivoice")
    providers = plugin._VoiceCapability(_Owner()).available_providers()
    assert "faster-whisper" not in providers["stt"]
    stt = providers["unavailable"]["stt"]["faster-whisper"]
    assert stt["code"] == "runtime_missing" and "faster_whisper" in stt["reason"]
    assert providers["unavailable"]["cloning"]["omnivoice"]["code"] == "runtime_missing"
