"""Provider credentials from the HOST, and discovery without them.

A host (AbstractCore, fed by a gateway's Providers screen) hands the voice plugin
an OpenAI key through settings, never an env var: `voice_openai_api_key` /
`voice_openai_base_url`. That key must reach OpenAI synthesis/STT/cloning and
never the OpenAI-compatible server. Without any key, the unfiltered
`voice_catalog()` explains why instead of raising.
"""

from __future__ import annotations

import pytest

import abstractvoice.integrations.abstractcore_plugin as plugin
from abstractvoice.adapters.openai_compatible_http import OPENAI_DEFAULT_BASE_URL, remote_endpoint

_ENV = (
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "ABSTRACTVOICE_TTS_ENGINE",
    "ABSTRACTVOICE_STT_ENGINE",
    "ABSTRACTGATEWAY_VOICE_TTS_ENGINE",
    "ABSTRACTGATEWAY_VOICE_STT_ENGINE",
    "ABSTRACTVOICE_CLONING_ENGINE",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in _ENV:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(plugin, "_VM_CACHE", {})


class _Owner:
    def __init__(self, config=None):
        self.config = dict(config or {})


# --- endpoint resolution ------------------------------------------------------


def test_openai_key_never_goes_to_the_compatible_server():
    kw = dict(remote_base_url="http://llm.local/v1", remote_api_key="local-key", openai_api_key="sk-host")
    assert remote_endpoint("openai", **kw) == (OPENAI_DEFAULT_BASE_URL, "sk-host")
    assert remote_endpoint("openai-compatible", **kw) == ("http://llm.local/v1", "local-key")
    assert remote_endpoint("openai", openai_api_key="sk-host", openai_base_url="https://eu.example/v1") == (
        "https://eu.example/v1",
        "sk-host",
    )


def test_host_key_ignores_openai_base_url_env(monkeypatch):
    """OPENAI_BASE_URL names the compatible server; a host OpenAI key never goes there.

    The autouse fixture deletes OPENAI_BASE_URL, so this test sets it again.
    """
    from abstractvoice.cloning.manager import VoiceCloner
    from abstractvoice.vm.manager import VoiceManager

    monkeypatch.setenv("OPENAI_BASE_URL", "http://compat.local:8000/v1")
    assert remote_endpoint("openai", openai_api_key="sk-HOST") == (OPENAI_DEFAULT_BASE_URL, "sk-HOST")

    vm = VoiceManager(tts_engine="openai", stt_engine="openai", openai_api_key="sk-HOST")
    tts = vm.tts_adapter
    assert (tts.base_url, tts.api_key) == (OPENAI_DEFAULT_BASE_URL, "sk-HOST")
    stt = vm._get_stt_adapter()
    assert (stt.base_url, stt.api_key) == (OPENAI_DEFAULT_BASE_URL, "sk-HOST")
    engine = VoiceCloner(openai_api_key="sk-HOST")._get_engine("openai")
    assert (engine.base_url, engine.api_key) == (OPENAI_DEFAULT_BASE_URL, "sk-HOST")


def test_without_openai_credentials_the_shared_endpoint_is_used():
    assert remote_endpoint("openai", remote_base_url=None, remote_api_key="sk-legacy") == (None, "sk-legacy")


# --- the VoiceManager hands each adapter its own credentials ------------------


def test_voice_manager_tts_and_stt_use_the_openai_credentials(monkeypatch):
    import abstractvoice.vm.manager as manager

    seen = {}

    def fake_create(**kwargs):
        seen.update(kwargs)
        return None, kwargs["engine"]

    monkeypatch.setattr(manager, "create_tts_adapter", fake_create)
    vm = manager.VoiceManager(
        tts_engine="openai",
        stt_engine="openai",
        remote_base_url="http://llm.local/v1",
        remote_api_key="local-key",
        openai_api_key="sk-host",
    )
    assert (seen["base_url"], seen["api_key"]) == (OPENAI_DEFAULT_BASE_URL, "sk-host")
    stt = vm._get_stt_adapter()
    assert stt is not None
    assert stt.api_key == "sk-host"
    assert "api.openai.com" in stt.base_url


def test_cloner_openai_engine_uses_the_openai_credentials():
    from abstractvoice.cloning.manager import VoiceCloner

    cloner = VoiceCloner(
        remote_base_url="http://llm.local/v1",
        remote_api_key="local-key",
        openai_api_key="sk-host",
    )
    engine = cloner._get_engine("openai")
    assert engine.api_key == "sk-host"
    assert engine.base_url == OPENAI_DEFAULT_BASE_URL
    compatible = cloner._get_engine("openai-compatible")
    assert compatible.api_key == "local-key"


# --- the plugin takes them from host settings ---------------------------------


def test_plugin_setting_makes_openai_available_and_reaches_the_manager(monkeypatch):
    import abstractvoice.voice_manager as voice_manager_module

    built = []

    class _VM:
        def __init__(self, **kwargs):
            built.append(kwargs)

    monkeypatch.setattr(voice_manager_module, "VoiceManager", _VM)
    cap = plugin._VoiceCapability(_Owner({"voice_openai_api_key": "sk-host"}))
    providers = cap.available_providers()
    assert "openai" in providers["tts"] and "openai" in providers["stt"]
    assert "openai" not in providers["unavailable"]["tts"]
    cap._get_vm()
    assert built[0]["openai_api_key"] == "sk-host"
    assert built[0]["openai_base_url"] is None


# --- no key: discovery explains instead of raising ----------------------------


def test_unfiltered_catalog_without_openai_key_returns_reasons():
    cap = plugin._VoiceCapability(_Owner({"voice_tts_engine": "openai"}))
    catalog = cap.voice_catalog()
    assert catalog["unavailable_providers"]["tts"]["openai"]["code"] == "not_configured"
    assert "openai" not in catalog["tts_providers"]
    if not catalog["tts_providers"]:
        assert "no OpenAI API key is configured" in catalog["unavailable_reason"]


def test_switching_tts_engine_and_cloning_keep_the_openai_credentials(monkeypatch):
    import abstractvoice.vm.manager as manager
    import abstractvoice.vm.tts_mixin as tts_mixin

    seen = []

    def fake_create(**kwargs):
        seen.append(kwargs)
        return None, kwargs["engine"]

    monkeypatch.setattr(manager, "create_tts_adapter", fake_create)
    monkeypatch.setattr(tts_mixin, "create_tts_adapter", fake_create)
    vm = manager.VoiceManager(
        tts_engine="openai-compatible",
        stt_engine="openai",
        remote_base_url="http://llm.local/v1",
        remote_api_key="local-key",
        openai_api_key="sk-host",
    )
    assert (seen[-1]["base_url"], seen[-1]["api_key"]) == ("http://llm.local/v1", "local-key")
    with pytest.raises(RuntimeError):
        vm.set_tts_engine("openai")  # the fake returns no adapter
    assert (seen[-1]["base_url"], seen[-1]["api_key"]) == (OPENAI_DEFAULT_BASE_URL, "sk-host")
    cloner = vm._get_voice_cloner()
    assert cloner._get_engine("openai").api_key == "sk-host"
