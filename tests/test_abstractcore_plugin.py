import pytest

from abstractvoice.integrations.abstractcore_plugin import _AudioCapability, _VoiceCapability, register


_PLUGIN_ENV_KEYS = (
    "ABSTRACTVOICE_LANGUAGE",
    "ABSTRACTVOICE_ALLOW_DOWNLOADS",
    "ABSTRACTVOICE_TTS_ENGINE",
    "ABSTRACTVOICE_STT_ENGINE",
    "ABSTRACTVOICE_TTS_MODEL",
    "ABSTRACTVOICE_STT_MODEL",
    "ABSTRACTVOICE_REMOTE_TIMEOUT_S",
    "ABSTRACTVOICE_OPENAI_TTS_MODEL",
    "ABSTRACTVOICE_OPENAI_STT_MODEL",
    "ABSTRACTVOICE_OPENAI_TIMEOUT_S",
    "ABSTRACTVOICE_OPENAI_COMPATIBLE_TTS_MODEL",
    "ABSTRACTVOICE_OPENAI_COMPATIBLE_STT_MODEL",
    "ABSTRACTVOICE_OPENAI_COMPATIBLE_TIMEOUT_S",
    "ABSTRACTVOICE_REMOTE_TTS_MODEL",
    "ABSTRACTVOICE_REMOTE_STT_MODEL",
    "ABSTRACTVOICE_CLONING_ENGINE",
    "ABSTRACTVOICE_CLONED_TTS_STREAMING",
    "ABSTRACTVOICE_TTS_DELIVERY_MODE",
    "ABSTRACTVOICE_DEBUG",
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
)


def _clear_plugin_env(monkeypatch):
    for key in _PLUGIN_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_register_adds_voice_and_audio_backends():
    calls = {"voice": None, "audio": None}

    class _Registry:
        def register_voice_backend(self, **kwargs):
            calls["voice"] = dict(kwargs)

        def register_audio_backend(self, **kwargs):
            calls["audio"] = dict(kwargs)

    register(_Registry())
    assert calls["voice"]["backend_id"] == "abstractvoice:default"
    assert callable(calls["voice"]["factory"])
    assert calls["audio"]["backend_id"] == "abstractvoice:stt"
    assert callable(calls["audio"]["factory"])


def test_voice_capability_defaults_to_openai_env_for_abstractcore(monkeypatch):
    import abstractvoice.integrations.abstractcore_plugin as plugin

    _clear_plugin_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    seen = {}

    class _VM:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    plugin._VM_CACHE.clear()
    monkeypatch.setattr("abstractvoice.voice_manager.VoiceManager", _VM)

    class _Owner:
        config = {}

    cap = _VoiceCapability(_Owner())
    assert cap._get_vm() is not None

    assert seen["tts_engine"] == "openai"
    assert seen["stt_engine"] == "openai"
    assert seen["cloning_engine"] == "omnivoice"
    assert seen["remote_api_key"] is None


def test_voice_capability_env_overrides_openai_defaults(monkeypatch):
    import abstractvoice.integrations.abstractcore_plugin as plugin

    _clear_plugin_env(monkeypatch)
    monkeypatch.setenv("ABSTRACTVOICE_TTS_ENGINE", "openai-compatible")
    monkeypatch.setenv("ABSTRACTVOICE_STT_ENGINE", "openai-compatible")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://remote.test/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "remote-key")
    monkeypatch.setenv("ABSTRACTVOICE_ALLOW_DOWNLOADS", "false")
    monkeypatch.setenv("ABSTRACTVOICE_CLONED_TTS_STREAMING", "off")
    monkeypatch.setenv("ABSTRACTVOICE_DEBUG", "true")
    seen = {}

    class _VM:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    plugin._VM_CACHE.clear()
    monkeypatch.setattr("abstractvoice.voice_manager.VoiceManager", _VM)

    class _Owner:
        config = {}

    cap = _VoiceCapability(_Owner())
    assert cap._get_vm() is not None

    assert seen["tts_engine"] == "openai-compatible"
    assert seen["stt_engine"] == "openai-compatible"
    assert seen["remote_base_url"] == "http://remote.test/v1"
    assert seen["remote_api_key"] is None
    assert seen["allow_downloads"] is False
    assert seen["cloning_engine"] == "omnivoice"
    assert seen["cloned_tts_streaming"] is False
    assert seen["debug_mode"] is True


def test_voice_capability_owner_config_overrides_openai_env(monkeypatch):
    import abstractvoice.integrations.abstractcore_plugin as plugin

    _clear_plugin_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    seen = {}

    class _VM:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    plugin._VM_CACHE.clear()
    monkeypatch.setattr("abstractvoice.voice_manager.VoiceManager", _VM)

    class _Owner:
        config = {
            "voice_tts_engine": "openai-compatible",
            "voice_stt_engine": "openai-compatible",
            "voice_remote_base_url": "http://remote.test/v1",
            "voice_remote_api_key": "remote-key",
        }

    cap = _VoiceCapability(_Owner())
    assert cap._get_vm() is not None

    assert seen["tts_engine"] == "openai-compatible"
    assert seen["stt_engine"] == "openai-compatible"
    assert seen["remote_base_url"] == "http://remote.test/v1"
    assert seen["remote_api_key"] == "remote-key"


def test_voice_capability_owner_config_string_bools_are_not_truthy(monkeypatch):
    import abstractvoice.integrations.abstractcore_plugin as plugin

    _clear_plugin_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    seen = {}

    class _VM:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    plugin._VM_CACHE.clear()
    monkeypatch.setattr("abstractvoice.voice_manager.VoiceManager", _VM)

    class _Owner:
        config = {
            "voice_allow_downloads": "false",
            "voice_cloned_tts_streaming": "0",
            "voice_tts_streaming": "false",
            "voice_debug_mode": "off",
        }

    cap = _VoiceCapability(_Owner())
    assert cap._get_vm() is not None

    assert seen["allow_downloads"] is False
    assert seen["cloned_tts_streaming"] is False
    assert seen["tts_delivery_mode"] == "buffered"
    assert seen["debug_mode"] is False


def test_voice_capability_default_openai_without_api_key_has_clear_error(monkeypatch):
    import abstractvoice.integrations.abstractcore_plugin as plugin

    _clear_plugin_env(monkeypatch)
    plugin._VM_CACHE.clear()

    class _Owner:
        config = {}

    cap = _VoiceCapability(_Owner())
    with pytest.raises(ValueError, match="OpenAI audio requires OPENAI_API_KEY"):
        cap._get_vm()


def test_voice_capability_available_providers_lists_available_engines_without_vm(monkeypatch):
    import abstractvoice.integrations.abstractcore_plugin as plugin

    _clear_plugin_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://remote.test/v1")
    monkeypatch.setenv("ABSTRACTVOICE_OPENAI_VOICE_CONSENT_ID", "consent-123")
    monkeypatch.setattr(plugin, "_local_tts_engine_available", lambda engine: plugin._norm_engine_id(engine) == "piper")
    monkeypatch.setattr(
        plugin,
        "_local_stt_engine_available",
        lambda engine: plugin._norm_engine_id(engine) in {"faster-whisper", "transformers-asr"},
    )
    monkeypatch.setattr(plugin, "_local_cloning_engine_available", lambda engine: plugin._norm_engine_id(engine) in {"omnivoice", "audiodit"})

    class _Owner:
        config = {}

    cap = _VoiceCapability(_Owner())
    providers = cap.available_providers()

    assert providers["active_tts_provider"] == "openai"
    assert providers["active_stt_provider"] == "openai"
    assert providers["active_cloning_provider"] == "omnivoice"
    assert providers["tts"] == ["openai", "openai-compatible", "piper"]
    assert providers["stt"] == ["openai", "openai-compatible", "faster-whisper", "transformers-asr"]
    assert providers["cloning"] == ["omnivoice", "audiodit", "openai", "openai-compatible"]
    assert "supertonic" in providers["known_tts_providers"]
    assert providers["details"]["tts"]["openai"]["remote"] is True
    assert providers["details"]["tts"]["piper"]["local"] is True
    assert cap.list_available_providers() == providers


def test_voice_capability_catalog_surface_serializes_profiles_and_models(monkeypatch):
    from abstractvoice.voice_profiles import VoiceProfile

    _clear_plugin_env(monkeypatch)
    monkeypatch.delenv("ABSTRACTGATEWAY_VOICE_TTS_ENGINE", raising=False)
    monkeypatch.delenv("ABSTRACTVOICE_OPENAI_TTS_MODELS", raising=False)
    monkeypatch.delenv("ABSTRACTVOICE_OPENAI_TTS_VOICE", raising=False)
    monkeypatch.delenv("ABSTRACTVOICE_OPENAI_TTS_VOICES", raising=False)
    monkeypatch.setattr(
        "abstractvoice.integrations.abstractcore_plugin._catalog_safe_local_tts_engines",
        lambda: [],
    )
    monkeypatch.setattr(
        "abstractvoice.integrations.abstractcore_plugin._local_stt_engine_available",
        lambda engine: str(engine or "").strip().lower().replace("_", "-") in {"faster-whisper", "transformers-asr"},
    )

    class _Adapter:
        engine_id = "openai"
        model_id = "gpt-active-tts"

    class _VM:
        tts_adapter = _Adapter()

        def get_profiles(self, *, kind="tts"):
            assert kind == "tts"
            return [
                VoiceProfile(
                    engine_id="openai",
                    profile_id="alloy",
                    label="Alloy",
                    params={"voice": "alloy"},
                    tags={"provider": "openai"},
                ),
                {
                    "engine_id": "openai",
                    "profile_id": "dict_voice",
                    "label": "Dict Voice",
                    "params": {"voice": "dict_voice"},
                },
            ]

        def get_active_profile(self, *, kind="tts"):
            assert kind == "tts"
            return VoiceProfile(
                engine_id="openai",
                profile_id="alloy",
                label="Alloy",
                params={"voice": "alloy"},
                tags={"provider": "openai"},
            )

        def list_available_models(self):
            return {
                "openai": {
                    "alloy": {
                        "model": "gpt-active-tts",
                        "available_models": ["gpt-active-tts", "tts-1"],
                        "remote": True,
                    },
                    "nova": {
                        "available_models": ["tts-1", "tts-1-hd"],
                        "remote": True,
                    },
                    "lessac": {
                        "model_filename": "en_US-lessac-medium.onnx",
                        "remote": False,
                    },
                }
            }

        stt_engine = "openai"
        stt_model = "gpt-active-stt"

        def list_cloned_voices(self):
            return [
                {
                    "voice_id": "clone_laurent",
                    "name": "Laurent",
                    "backend": "xtts",
                    "metadata": {"speaker_wav": "laurent.wav"},
                }
            ]

    class _Owner:
        def __init__(self):
            self.config = {"voice_manager_instance": _VM()}

    cap = _VoiceCapability(_Owner())

    profiles = cap.list_profiles()
    assert [p["profile_id"] for p in profiles] == ["alloy", "dict_voice"]
    assert profiles[0]["params"] == {"voice": "alloy"}

    assert cap.list_tts_models() == ["gpt-active-tts", "tts-1", "tts-1-hd", "en_US-lessac-medium.onnx"]
    expected_stt_models = ["gpt-active-stt", "gpt-4o-transcribe", "gpt-4o-mini-transcribe", "whisper-1"]
    assert cap.list_stt_models()[: len(expected_stt_models)] == expected_stt_models

    catalog = cap.voice_catalog()
    assert catalog["engine_id"] == "openai"
    assert catalog["active_profile"]["profile_id"] == "alloy"
    assert catalog["active_model"] == "gpt-active-tts"
    assert catalog["active_tts_provider"] == "openai"
    assert catalog["active_stt_provider"] == "openai"
    assert catalog["tts_providers"] == ["openai"]
    assert catalog["stt_providers"][0] == "openai"
    assert "openai" in catalog["available_tts_providers"]
    assert "supertonic" in catalog["available_providers"]["known_tts_providers"]
    assert "piper" in catalog["available_providers"]["known_tts_providers"]
    assert "audiodit" in catalog["available_providers"]["known_tts_providers"]
    assert "omnivoice" in catalog["available_providers"]["known_tts_providers"]
    assert "openai" in catalog["available_stt_providers"]
    assert "transformers-asr" in catalog["available_providers"]["known_stt_providers"]
    assert "available_providers" in catalog
    assert catalog["tts_models"] == ["gpt-active-tts", "tts-1", "tts-1-hd", "en_US-lessac-medium.onnx"]
    assert catalog["stt_models"][: len(expected_stt_models)] == expected_stt_models
    assert "transformers-asr" in catalog["stt_providers"]
    assert "transformers-asr" in catalog["stt_models_by_provider"]
    assert "openai/whisper-large-v3" in catalog["stt_models_by_provider"]["transformers-asr"]
    assert "Qwen/Qwen3-ASR-1.7B" in catalog["stt_models_by_provider"]["transformers-asr"]
    assert catalog["cloned_voices"][0]["voice_id"] == "clone_laurent"
    assert catalog["voices"][-1]["kind"] == "clone"
    assert catalog["catalog"]["openai"]["alloy"]["remote"] is True
    assert catalog["tts_catalog_by_provider"]["openai"]["models"] == ["gpt-active-tts", "tts-1", "tts-1-hd", "en_US-lessac-medium.onnx"]
    assert catalog["tts_catalog_by_provider"]["openai"]["model_variants"][0] == "openai"
    assert "openai:gpt-active-tts" in catalog["tts_catalog_by_provider"]["openai"]["model_variants"]
    assert "alloy" in {item["profile_id"] for item in catalog["tts_catalog_by_provider"]["openai"]["profiles"]}
    assert "clone_laurent" in {item["voice_id"] for item in catalog["tts_catalog_by_provider"]["openai"]["cloned_voices"]}
    assert "gpt-active-stt" in catalog["stt_catalog_by_provider"]["openai"]["models"]
    assert "openai:gpt-4o-transcribe" in catalog["stt_catalog_by_provider"]["openai"]["model_variants"]


def test_voice_capability_catalog_controls_reflect_vm_tts_capabilities():
    from abstractvoice.speech_request import SpeechCapabilities, SpeechCapability

    class _Adapter:
        engine_id = "audiodit"
        model_id = "dramatic-local"

    class _VM:
        tts_adapter = _Adapter()
        stt_engine = "faster-whisper"

        def get_profiles(self, *, kind="tts"):
            assert kind == "tts"
            return []

        def list_available_models(self):
            return {}

        def get_tts_capabilities(self):
            return SpeechCapabilities(
                fields={
                    "speed": SpeechCapability(
                        name="speed",
                        support="unsupported",
                        reason="speed changes are disabled for this engine",
                    ),
                    "pace": SpeechCapability(
                        name="pace",
                        support="unsupported",
                        reason="directed-speech planning not wired yet",
                    ),
                }
            )

    class _Owner:
        config = {"voice_manager_instance": _VM()}

    catalog = _VoiceCapability(_Owner()).voice_catalog()

    assert catalog["speech_request_contract"] == "speech_request_v1"
    assert "compatibility_catalog" in catalog
    assert catalog["controls"]["speed"] == {"supported": True, "min": 0.5, "max": 2.0, "default": 1.0}
    assert catalog["controls"]["quality_preset"]["supported"] is True
    assert "pace" not in catalog["controls"]
    assert catalog["tts_capabilities"]["speed"]["support"] == "unsupported"
    assert catalog["tts_capabilities"]["speed"]["reason"] == "speed changes are disabled for this engine"
    assert catalog["tts_capabilities"]["pace"]["support"] == "unsupported"
    assert catalog["tts_capabilities"]["pace"]["reason"] == "directed-speech planning not wired yet"


def test_provider_filtered_voice_catalog_uses_light_metadata_without_vm(monkeypatch):
    import abstractvoice.integrations.abstractcore_plugin as plugin

    _clear_plugin_env(monkeypatch)
    monkeypatch.setattr(plugin, "_local_tts_engines", lambda: ["omnivoice"])
    monkeypatch.setattr(plugin, "_local_tts_engine_available", lambda engine: plugin._norm_engine_id(engine) == "omnivoice")
    monkeypatch.setattr(plugin, "_local_cloning_engine_available", lambda engine: plugin._norm_engine_id(engine) == "omnivoice")

    class _Store:
        def list_voices(self):
            return [{"voice_id": "clone-a", "name": "Clone A", "engine": "omnivoice"}]

    monkeypatch.setattr("abstractvoice.cloning.store.VoiceCloneStore", _Store)

    class _Owner:
        config = {}

    cap = _VoiceCapability(_Owner())
    cap._get_vm = lambda: (_ for _ in ()).throw(AssertionError("provider-filtered catalog must not initialize VoiceManager"))

    catalog = cap.voice_catalog(provider="omnivoice")

    assert catalog["source"] == "abstractvoice.light_catalog"
    assert catalog["tts_providers"] == ["omnivoice"]
    assert "default" in {item.get("profile_id") for item in catalog["profiles"]}
    assert "clone-a" in {item.get("voice_id") for item in catalog["cloned_voices"]}
    assert {"en", "fr"} <= set(catalog["tts_models_by_provider"]["omnivoice"])
    assert catalog["tts_model_roles_by_provider"]["omnivoice"] == "language"
    assert {"en", "fr"} <= set(cap.list_tts_models(provider="omnivoice"))


def test_voice_catalog_filters_unavailable_local_stt_providers(monkeypatch):
    _clear_plugin_env(monkeypatch)
    monkeypatch.setattr(
        "abstractvoice.integrations.abstractcore_plugin._local_stt_engine_available",
        lambda _engine: False,
    )

    class _Adapter:
        engine_id = "openai"
        model_id = "gpt-active-tts"

    class _VM:
        tts_adapter = _Adapter()
        stt_engine = "openai"
        stt_model = "gpt-active-stt"

        def get_profiles(self, *, kind="tts"):
            assert kind == "tts"
            return []

        def list_available_models(self):
            return {}

    class _Owner:
        config = {"voice_manager_instance": _VM()}

    catalog = _VoiceCapability(_Owner()).voice_catalog()

    assert catalog["stt_providers"] == ["openai"]
    assert "faster-whisper" not in catalog["stt_models_by_provider"]
    assert "transformers-asr" not in catalog["stt_models_by_provider"]
    assert "transformers-asr" in catalog["available_providers"]["known_stt_providers"]


def test_voice_catalog_does_not_treat_unloaded_local_stt_engine_name_as_available(monkeypatch):
    _clear_plugin_env(monkeypatch)
    monkeypatch.setattr(
        "abstractvoice.integrations.abstractcore_plugin._local_stt_engine_available",
        lambda _engine: False,
    )

    class _Adapter:
        engine_id = "openai"
        model_id = "gpt-active-tts"

    class _VM:
        tts_adapter = _Adapter()
        stt_engine = "transformers-asr"
        stt_model = "openai/whisper-large-v3"

        def get_profiles(self, *, kind="tts"):
            assert kind == "tts"
            return []

        def list_available_models(self):
            return {}

    class _Owner:
        config = {"voice_manager_instance": _VM()}

    cap = _VoiceCapability(_Owner())
    catalog = cap.voice_catalog()
    models = cap.list_stt_models()

    assert catalog["available_stt_providers"] == []
    assert "transformers-asr" not in catalog["stt_providers"]
    assert "transformers-asr" not in catalog["stt_models_by_provider"]
    assert "openai/whisper-large-v3" in models
    assert "Qwen/Qwen3-ASR-1.7B" not in models


def test_voice_capability_tts_forwards_instructions_to_vm():
    seen: dict[str, object] = {}

    class _VM:
        def speak_to_bytes(self, text: str, format: str = "wav", voice=None, instructions=None):
            seen.update(
                {
                    "text": text,
                    "format": format,
                    "voice": voice,
                    "instructions": instructions,
                }
            )
            return b"RIFF....WAVE"

    class _Owner:
        config = {"voice_manager_instance": _VM()}

    out = _VoiceCapability(_Owner()).tts(
        "hello",
        instructions="speak softly",
    )

    assert out.startswith(b"RIFF")
    assert seen == {
        "text": "hello",
        "format": "wav",
        "voice": None,
        "instructions": "speak softly",
    }


def test_voice_capability_provider_filtered_model_and_voice_discovery(monkeypatch):
    from abstractvoice.voice_profiles import VoiceProfile

    _clear_plugin_env(monkeypatch)
    monkeypatch.delenv("ABSTRACTGATEWAY_VOICE_TTS_ENGINE", raising=False)
    monkeypatch.delenv("ABSTRACTVOICE_OPENAI_TTS_MODELS", raising=False)
    monkeypatch.setattr(
        "abstractvoice.integrations.abstractcore_plugin._catalog_safe_local_tts_engines",
        lambda: [],
    )

    class _Adapter:
        engine_id = "openai"
        model_id = "gpt-active-tts"

    class _VM:
        tts_adapter = _Adapter()
        stt_engine = "openai"
        stt_model = "gpt-active-stt"

        def get_profiles(self, *, kind="tts"):
            assert kind == "tts"
            return [
                VoiceProfile(
                    engine_id="openai",
                    profile_id="alloy",
                    label="Alloy",
                    params={"voice": "alloy"},
                    tags={"provider": "openai"},
                ),
                {
                    "engine_id": "piper",
                    "profile_id": "amy",
                    "label": "Amy",
                    "params": {"voice": "amy", "model": "en_US-amy-medium.onnx"},
                },
            ]

        def list_available_models(self):
            return {
                "openai": {
                    "alloy": {"available_models": ["gpt-active-tts", "tts-1"], "remote": True},
                },
                "en": {
                    "amy": {"model_filename": "en_US-amy-medium.onnx", "cached": True},
                },
            }

        def list_cloned_voices(self):
            return [
                {"voice_id": "clone_openai", "name": "OpenAI Clone", "engine": "openai"},
                {"voice_id": "clone_piper", "name": "Piper Clone", "engine": "piper"},
            ]

    class _Owner:
        config = {"voice_manager_instance": _VM()}

    cap = _VoiceCapability(_Owner())

    assert cap.list_models(kind="tts", provider="openai") == ["gpt-active-tts", "tts-1"]
    assert cap.list_tts_models(provider="piper") == ["en_US-amy-medium.onnx"]
    assert cap.list_models(kind="stt", provider="openai")[0] == "gpt-active-stt"

    openai_voices = cap.list_tts_voices(provider="openai", model="tts-1")
    assert {item.get("profile_id") or item.get("voice_id") for item in openai_voices} >= {"alloy", "clone_openai"}
    assert "clone_piper" not in {item.get("voice_id") for item in openai_voices}

    piper_voices = cap.list_voices(provider="piper", model="en_US-amy-medium.onnx")
    assert {item.get("profile_id") or item.get("voice_id") for item in piper_voices} >= {"amy", "clone_piper"}
    assert "alloy" not in {item.get("profile_id") for item in piper_voices}

    piper_no_clones = cap.list_tts_voices(provider="piper", model="en_US-amy-medium.onnx", include_clones=False)
    assert {item.get("profile_id") for item in piper_no_clones} == {"amy"}
    assert cap.list_cloned_voices(provider="openai", model="gpt-active-tts")[0]["voice_id"] == "clone_openai"


def test_voice_capability_list_models_supports_cloning_kind(monkeypatch):
    from abstractvoice.compatibility import build_compatibility_catalog

    _clear_plugin_env(monkeypatch)

    class _VM:
        def get_compatibility_catalog(self):
            return build_compatibility_catalog(
                current_cloning_provider="openai-compatible",
                current_remote_tts_model="custom-compatible-tts",
            )

    class _Owner:
        config = {"voice_manager_instance": _VM()}

    cap = _VoiceCapability(_Owner())

    assert cap.list_models(kind="cloning", provider="f5_tts") == ["mrfakename/OpenF5-TTS-Base"]
    assert cap.list_cloning_models(provider="f5_tts") == ["mrfakename/OpenF5-TTS-Base"]
    assert cap.list_models(kind="cloning", provider="openai-compatible:custom-compatible-tts") == ["custom-compatible-tts"]


def test_voice_capability_cloning_query_helpers_delegate_to_vm(monkeypatch):
    _clear_plugin_env(monkeypatch)

    seen = {"support": None, "matches": None}

    class _VM:
        def get_capability_support(self, **kwargs):
            seen["support"] = dict(kwargs)
            return {"support": "native", "reason": "sentinel"}

        def find_compatible_models(self, **kwargs):
            seen["matches"] = dict(kwargs)
            return [
                {
                    "kind": "cloning",
                    "provider": "f5_tts",
                    "model": "mrfakename/OpenF5-TTS-Base",
                    "surface": "create",
                    "feature": "multi_reference_audio",
                    "support": {"support": "native", "reason": None, "metadata": {}},
                }
            ]

    class _Owner:
        config = {"voice_manager_instance": _VM()}

    cap = _VoiceCapability(_Owner())

    support = cap.get_capability_support(
        kind="cloning",
        provider="f5_tts",
        model="mrfakename/OpenF5-TTS-Base",
        surface="create",
        feature="multi_reference_audio",
    )
    matches = cap.find_compatible_models(
        kind="cloning",
        feature="multi_reference_audio",
        surface="create",
        support_in="native",
    )

    assert support == {"support": "native", "reason": "sentinel"}
    assert matches[0]["provider"] == "f5_tts"
    assert seen["support"] == {
        "kind": "cloning",
        "feature": "multi_reference_audio",
        "provider": "f5_tts",
        "model": "mrfakename/OpenF5-TTS-Base",
        "surface": "create",
    }
    assert seen["matches"] == {
        "kind": "cloning",
        "feature": "multi_reference_audio",
        "surface": "create",
        "support_in": ("native",),
    }


def test_voice_capability_cloning_query_helpers_fallback_to_catalog(monkeypatch):
    from abstractvoice.compatibility import build_compatibility_catalog

    _clear_plugin_env(monkeypatch)

    class _VM:
        def get_compatibility_catalog(self):
            return build_compatibility_catalog()

    class _Owner:
        config = {"voice_manager_instance": _VM()}

    cap = _VoiceCapability(_Owner())

    support = cap.get_capability_support(
        kind="cloning",
        provider="omnivoice",
        model="k2-fsa/OmniVoice",
        surface="create",
        feature="multi_reference_audio",
    )
    matches = cap.find_compatible_models(
        kind="cloning",
        feature="multi_reference_audio",
        surface="create",
        support_in="native",
    )

    assert support is not None
    assert support["support"] == "unsupported"
    providers = {item["provider"] for item in matches}
    assert "f5_tts" in providers
    assert "audiodit" in providers
    assert "omnivoice" not in providers


def test_voice_capability_clone_exposes_first_class_plugin_surface(tmp_path):
    seen = {"path": None, "kwargs": None}
    ref = tmp_path / "reference.wav"
    ref.write_bytes(b"RIFF....WAVE")

    class _VM:
        cloning_engine = "f5_tts"

        def clone_voice(self, path, **kwargs):
            seen["path"] = path
            seen["kwargs"] = dict(kwargs)
            return "clone-voice-id"

    class _Owner:
        config = {"voice_manager_instance": _VM()}

    cap = _VoiceCapability(_Owner())

    assert cap.clone(str(ref), name="Ada", reference_text="hello", engine="f5_tts") == "clone-voice-id"
    assert cap.clone_voice(str(ref), name="Ada", reference_text="hello", engine="f5_tts") == "clone-voice-id"
    assert seen["path"] == str(ref)
    assert seen["kwargs"] == {
        "name": "Ada",
        "reference_text": "hello",
        "engine": "f5_tts",
    }


def test_voice_capability_clone_routes_remote_provider_model_to_vm(monkeypatch):
    import abstractvoice.integrations.abstractcore_plugin as plugin

    _clear_plugin_env(monkeypatch)
    seen = {"init": None, "clone": None}

    class _VM:
        def __init__(self, **kwargs):
            seen["init"] = dict(kwargs)

        def clone_voice_from_wav_bytes(self, audio, **kwargs):
            seen["clone"] = {
                "audio": bytes(audio),
                "kwargs": dict(kwargs),
            }
            return "clone-voice-id"

    plugin._VM_CACHE.clear()
    monkeypatch.setattr("abstractvoice.voice_manager.VoiceManager", _VM)

    class _Owner:
        config = {}

    out = _VoiceCapability(_Owner()).clone(
        b"RIFF....WAVE",
        provider="openai-compatible:custom-compatible-tts",
        metadata={"source": "upload"},
    )

    assert out == "clone-voice-id"
    assert seen["init"]["tts_engine"] == "openai-compatible"
    assert seen["init"]["cloning_engine"] == "openai-compatible"
    assert seen["init"]["tts_model"] == "custom-compatible-tts"
    assert seen["clone"] == {
        "audio": b"RIFF....WAVE",
        "kwargs": {
            "name": None,
            "reference_text": None,
            "engine": "openai-compatible",
            "meta": {"source": "upload"},
        },
    }


def test_voice_capability_clone_does_not_force_local_cloning_model_into_tts(monkeypatch):
    import abstractvoice.integrations.abstractcore_plugin as plugin

    _clear_plugin_env(monkeypatch)
    seen = {"init": None}

    class _VM:
        def __init__(self, **kwargs):
            seen["init"] = dict(kwargs)

        def clone_voice_from_wav_bytes(self, audio, **kwargs):
            return "clone-voice-id"

    plugin._VM_CACHE.clear()
    monkeypatch.setattr("abstractvoice.voice_manager.VoiceManager", _VM)

    class _Owner:
        config = {}

    out = _VoiceCapability(_Owner()).clone(
        b"RIFF....WAVE",
        provider="f5_tts:mrfakename/OpenF5-TTS-Base",
    )

    assert out == "clone-voice-id"
    assert seen["init"]["cloning_engine"] == "f5_tts"
    assert seen["init"].get("tts_model") in (None, "")


def test_voice_capability_clone_uses_temp_file_for_named_audio_payload_dict(monkeypatch):
    import abstractvoice.integrations.abstractcore_plugin as plugin

    _clear_plugin_env(monkeypatch)
    seen = {"path": None, "clone_from_bytes_called": False}

    class _VM:
        def __init__(self, **kwargs):
            pass

        def clone_voice(self, path, **kwargs):
            seen["path"] = str(path)
            return "clone-voice-id"

        def clone_voice_from_wav_bytes(self, audio, **kwargs):
            seen["clone_from_bytes_called"] = True
            raise AssertionError("expected temp-path flow for filename-bearing audio payload dict")

    plugin._VM_CACHE.clear()
    monkeypatch.setattr("abstractvoice.voice_manager.VoiceManager", _VM)

    class _Owner:
        config = {}

    out = _VoiceCapability(_Owner()).clone(
        {
            "content": b"ID3\x00\x00",
            "filename": "reference.mp3",
            "mime_type": "audio/mpeg",
        },
        provider="openai-compatible",
    )

    assert out == "clone-voice-id"
    assert seen["clone_from_bytes_called"] is False
    assert isinstance(seen["path"], str) and seen["path"].endswith(".mp3")


def test_voice_catalog_keeps_remote_clones_when_base_tts_is_local(monkeypatch):
    from abstractvoice.voice_profiles import VoiceProfile

    _clear_plugin_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("ABSTRACTVOICE_OPENAI_VOICE_CREATE_PATH", "/v1/voices")
    monkeypatch.setattr(
        "abstractvoice.integrations.abstractcore_plugin._catalog_safe_local_tts_engines",
        lambda: [],
    )

    class _Adapter:
        engine_id = "piper"
        model_id = "en_US-amy-medium.onnx"

    class _VM:
        tts_adapter = _Adapter()
        stt_engine = "openai"
        stt_model = "gpt-active-stt"

        def get_profiles(self, *, kind="tts"):
            assert kind == "tts"
            return [
                VoiceProfile(
                    engine_id="piper",
                    profile_id="amy",
                    label="Amy",
                    params={"voice": "amy", "model": "en_US-amy-medium.onnx"},
                ),
            ]

        def list_available_models(self):
            return {
                "en": {
                    "amy": {"model_filename": "en_US-amy-medium.onnx", "cached": True},
                }
            }

        def list_cloned_voices(self):
            return [
                {"voice_id": "clone_openai", "name": "OpenAI Clone", "engine": "openai"},
            ]

    class _Owner:
        config = {"voice_manager_instance": _VM()}

    catalog = _VoiceCapability(_Owner()).voice_catalog()

    assert "openai" in catalog["available_cloning_providers"]
    assert "clone_openai" in {item["voice_id"] for item in catalog["cloned_voices"]}
    assert "clone_openai" in {
        item["voice_id"]
        for item in catalog["tts_catalog_by_provider"]["openai"]["cloned_voices"]
    }


def test_voice_capability_catalog_surfaces_stt_alias_variants(monkeypatch):
    _clear_plugin_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(
        "abstractvoice.integrations.abstractcore_plugin._local_stt_engine_available",
        lambda engine: str(engine or "").strip().lower().replace("_", "-") in {"faster-whisper", "transformers-asr"},
    )

    class _Owner:
        config = {}

    cap = _VoiceCapability(_Owner())
    models = cap.list_stt_models()

    assert "large" in models
    assert "large-v2" in models
    assert "large-v3" in models
    assert "whisper-large-v3" in models
    assert "whisper-large-v3-turbo" in models
    assert "qwen3-asr-1.7b" in models

    catalog = cap.voice_catalog()
    assert "large" in catalog["stt_models_by_provider"]["faster-whisper"]
    assert "large-v3" in catalog["stt_models_by_provider"]["faster-whisper"]
    assert "whisper-large-v3" in catalog["stt_models_by_provider"]["transformers-asr"]
    assert "whisper-large-v3-turbo" in catalog["stt_models_by_provider"]["transformers-asr"]
    assert "qwen3-asr-1.7b" in catalog["stt_models_by_provider"]["transformers-asr"]
    assert "faster-whisper:large" in catalog["stt_engine_variants"]["faster-whisper"]
    assert "transformers-asr:whisper-large-v3-turbo" in catalog["stt_engine_variants"]["transformers-asr"]
    assert "transformers-asr:qwen3-asr-1.7b" in catalog["stt_engine_variants"]["transformers-asr"]


def test_voice_capability_stt_accepts_provider_model_variants(monkeypatch):
    import abstractvoice.integrations.abstractcore_plugin as plugin

    _clear_plugin_env(monkeypatch)
    plugin._VM_CACHE.clear()
    created: list[dict[str, object]] = []

    class _VM:
        def __init__(self, **kwargs):
            created.append(dict(kwargs))
            self.whisper_model = kwargs.get("whisper_model", "base")
            self.stt_model = kwargs.get("stt_model")
            self._stt_engine_preference = kwargs.get("stt_engine", "openai")
            self.stt_adapter = None

        def transcribe_from_bytes(self, audio_bytes: bytes, language=None):
            return f"{self._stt_engine_preference}|{self.stt_model}|{self.whisper_model}"

    monkeypatch.setattr("abstractvoice.voice_manager.VoiceManager", _VM)

    class _Owner:
        config = {}

    cap = _VoiceCapability(_Owner())
    out_fw = cap.stt(b"audio", provider="faster-whisper:large")
    out_hf = cap.stt(b"audio", provider="transformers-asr:Qwen/Qwen3-ASR-1.7B")

    assert out_fw == "faster-whisper|large|large"
    assert out_hf == "transformers-asr|Qwen/Qwen3-ASR-1.7B|base"
    assert any(item["stt_engine"] == "faster-whisper" and item["whisper_model"] == "large" for item in created)
    assert any(
        item["stt_engine"] == "transformers-asr" and item["stt_model"] == "Qwen/Qwen3-ASR-1.7B"
        for item in created
    )


def test_voice_capability_tts_accepts_provider_model_variants(monkeypatch):
    import abstractvoice.integrations.abstractcore_plugin as plugin

    _clear_plugin_env(monkeypatch)
    plugin._VM_CACHE.clear()
    created: list[dict[str, object]] = []

    class _VM:
        def __init__(self, **kwargs):
            created.append(dict(kwargs))
            self.tts_model = kwargs.get("tts_model")
            self._abstractvoice_tts_engine = kwargs.get("tts_engine", "openai")
            self._tts_engine_name = kwargs.get("tts_engine", "openai")
            self.tts_adapter = None

        def speak_to_bytes(self, text: str, format: str = "wav", voice=None):
            return f"{self._tts_engine_name}|{self.tts_model}|{voice or ''}".encode("utf-8")

    monkeypatch.setattr("abstractvoice.voice_manager.VoiceManager", _VM)

    class _Owner:
        config = {}

    cap = _VoiceCapability(_Owner())
    out = cap.tts("hello", provider="openai:tts-1", voice="alloy")

    assert out == b"openai|tts-1|alloy"
    assert any(item["tts_engine"] == "openai" and item["tts_model"] == "tts-1" for item in created)


def test_voice_capability_catalog_omits_supertonic_profiles_without_runtime(monkeypatch):
    import abstractvoice.integrations.abstractcore_plugin as plugin
    from abstractvoice.voice_profiles import VoiceProfile

    _clear_plugin_env(monkeypatch)
    monkeypatch.setattr(plugin, "_local_tts_engines", lambda: ["supertonic"])
    monkeypatch.setattr(plugin, "_engine_runtime_available", lambda *_args, **_kwargs: False)

    class _Adapter:
        engine_id = "openai"
        model_id = "gpt-active-tts"

    class _VM:
        tts_adapter = _Adapter()
        stt_engine = "openai"
        stt_model = "gpt-active-stt"

        def get_profiles(self, *, kind="tts"):
            assert kind == "tts"
            return [
                VoiceProfile(
                    engine_id="openai",
                    profile_id="alloy",
                    label="Alloy",
                    params={"voice": "alloy"},
                    tags={"provider": "openai"},
                )
            ]

        def list_available_models(self):
            return {"openai": {"alloy": {"available_models": ["gpt-active-tts"], "remote": True}}}

    class _Owner:
        config = {"voice_manager_instance": _VM()}

    cap = _VoiceCapability(_Owner())
    cap._get_vm_for_provider = lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("runtime unavailable"))

    catalog = cap.voice_catalog()

    supertonic_ids = {
        item.get("profile_id")
        for item in catalog["profiles"]
        if item.get("provider") == "supertonic" or item.get("engine_id") == "supertonic"
    }
    assert supertonic_ids == set()
    assert "supertonic" not in catalog["tts_providers"]
    assert "supertonic" not in catalog["tts_voices_by_provider"]
    assert "supertonic" not in catalog["tts_models_by_provider"]


def test_voice_capability_catalog_includes_configured_openai_when_piper_active(monkeypatch):
    from abstractvoice.voice_profiles import VoiceProfile

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    class _PiperAdapter:
        engine_id = "piper"
        model_id = "en_US-amy-medium"

    class _PiperVM:
        tts_adapter = _PiperAdapter()

        def get_profiles(self, *, kind="tts"):
            assert kind == "tts"
            return [
                VoiceProfile(
                    engine_id="piper",
                    profile_id="amy",
                    label="Piper amy",
                    params={"voice": "amy", "model": "en_US-amy-medium"},
                    tags={"provider": "piper"},
                )
            ]

        def get_active_profile(self, *, kind="tts"):
            return self.get_profiles(kind=kind)[0]

        def list_available_models(self):
            return {"en": {"amy": {"model_filename": "en_US-amy-medium", "cached": True}}}

        stt_engine = "openai"

    class _OpenAIAdapter:
        engine_id = "openai"
        model_id = "gpt-4o-mini-tts"

    class _OpenAIVM:
        tts_adapter = _OpenAIAdapter()

        def get_profiles(self, *, kind="tts"):
            assert kind == "tts"
            return [
                VoiceProfile(
                    engine_id="openai",
                    profile_id="alloy",
                    label="Alloy",
                    params={"voice": "alloy"},
                    tags={"provider": "openai"},
                )
            ]

        def list_available_models(self):
            return {"openai": {"alloy": {"available_models": ["gpt-4o-mini-tts"], "remote": True}}}

    class _Owner:
        config = {"voice_manager_instance": _PiperVM()}

    cap = _VoiceCapability(_Owner())
    cap._get_vm_for_provider = lambda **kwargs: _OpenAIVM() if kwargs.get("tts_provider") == "openai" else _PiperVM()

    catalog = cap.voice_catalog()

    assert catalog["active_tts_provider"] == "piper"
    assert "piper" in catalog["tts_providers"]
    assert "openai" in catalog["tts_providers"]
    assert "gpt-4o-mini-tts" in catalog["tts_models"]
    ids = {p.get("profile_id") for p in catalog["profiles"]}
    assert {"amy", "alloy"} <= ids


def test_voice_capability_model_lists_survive_unavailable_default_vm(monkeypatch):
    _clear_plugin_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    class _Owner:
        config = {}

    cap = _VoiceCapability(_Owner())

    def unavailable_vm():
        raise RuntimeError("default voice manager unavailable")

    cap._get_vm = unavailable_vm

    assert "tts-1" in cap.list_tts_models()
    assert "whisper-1" in cap.list_stt_models()


def test_voice_capability_tts_treats_active_profile_voice_as_profile_not_clone():
    from abstractvoice.voice_profiles import VoiceProfile

    calls = {"profiles": [], "speak_voice": []}

    class _VM:
        def get_active_profile(self, *, kind="tts"):
            return VoiceProfile(engine_id="piper", profile_id="old", label="Old")

        def set_profile(self, profile_id: str, *, kind="tts"):
            calls["profiles"].append(profile_id)
            return profile_id in {"amy", "old"}

        def speak_to_bytes(self, text: str, format: str = "wav", voice=None):
            calls["speak_voice"].append(voice)
            return b"RIFF....WAVE"

    class _Owner:
        config = {"voice_manager_instance": _VM()}

    out = _VoiceCapability(_Owner()).tts("hello", voice="amy")

    assert out.startswith(b"RIFF")
    assert calls["profiles"] == ["amy", "old"]
    assert calls["speak_voice"] == [None]


def test_voice_capability_tts_does_not_apply_cloned_voice_as_profile():
    from abstractvoice.voice_profiles import VoiceProfile

    calls = {"profiles": [], "speak_voice": [], "clone_quality": []}

    class _VM:
        def get_active_profile(self, *, kind="tts"):
            return VoiceProfile(engine_id="omnivoice", profile_id="default", label="Default")

        def set_profile(self, profile_id: str, *, kind="tts"):
            calls["profiles"].append(profile_id)
            return True

        def get_cloned_voice(self, voice_id: str):
            if voice_id == "clone-1":
                return {"voice_id": "clone-1", "engine": "omnivoice"}
            return None

        def get_cloned_tts_quality_preset(self):
            return "standard"

        def set_cloned_tts_quality(self, preset: str):
            calls["clone_quality"].append(preset)
            return True

        def speak_to_bytes(self, text: str, format: str = "wav", voice=None):
            calls["speak_voice"].append(voice)
            return b"RIFF....WAVE"

    class _Owner:
        config = {"voice_manager_instance": _VM()}

    out = _VoiceCapability(_Owner()).tts("hello", voice="clone-1", profile="clone-1", quality_preset="high")

    assert out.startswith(b"RIFF")
    assert calls["profiles"] == []
    assert calls["clone_quality"] == ["high", "standard"]
    assert calls["speak_voice"] == ["clone-1"]


def test_voice_capability_tts_applies_explicit_profile_when_not_clone():
    from abstractvoice.voice_profiles import VoiceProfile

    calls = {"profiles": [], "speak_voice": []}

    class _VM:
        def get_active_profile(self, *, kind="tts"):
            return VoiceProfile(engine_id="omnivoice", profile_id="default", label="Default")

        def set_profile(self, profile_id: str, *, kind="tts"):
            calls["profiles"].append(profile_id)
            return profile_id in {"female_01", "default"}

        def get_cloned_voice(self, voice_id: str):
            return None

        def speak_to_bytes(self, text: str, format: str = "wav", voice=None):
            calls["speak_voice"].append(voice)
            return b"RIFF....WAVE"

    class _Owner:
        config = {"voice_manager_instance": _VM()}

    out = _VoiceCapability(_Owner()).tts("hello", profile="female_01")

    assert out.startswith(b"RIFF")
    assert calls["profiles"] == ["female_01", "default"]
    assert calls["speak_voice"] == [None]


def test_voice_capability_tts_treats_omnivoice_model_as_language():
    calls = {"provider_requests": [], "languages": [], "speak": []}

    class _Adapter:
        engine_id = "omnivoice"
        model_id = "k2-fsa/OmniVoice"

    class _VM:
        language = "en"
        tts_model = None
        tts_adapter = _Adapter()
        _abstractvoice_tts_engine = "omnivoice"

        def set_language(self, language: str):
            calls["languages"].append(language)
            self.language = language
            return True

        def speak_to_bytes(self, text: str, format: str = "wav", voice=None):
            calls["speak"].append({"text": text, "format": format, "voice": voice, "language": self.language})
            return b"RIFF....WAVE"

    class _Owner:
        config = {}

    vm = _VM()
    cap = _VoiceCapability(_Owner())

    def _get_vm_for_provider(**kwargs):
        calls["provider_requests"].append(dict(kwargs))
        return vm

    cap._get_vm_for_provider = _get_vm_for_provider
    out = cap.tts("bonjour", provider="omnivoice", model="fr")

    assert out.startswith(b"RIFF")
    assert calls["provider_requests"] == [{"tts_provider": "omnivoice", "tts_model": "fr"}]
    assert calls["languages"][0] == "fr"
    assert calls["speak"] == [{"text": "bonjour", "format": "wav", "voice": None, "language": "fr"}]
    assert vm.tts_adapter.model_id == "k2-fsa/OmniVoice"
    assert vm.tts_model is None


def test_voice_capability_injection_bytes_and_artifact():
    class _VM:
        def speak_to_bytes(self, text: str, format: str = "wav", voice=None):
            return b"RIFF....WAVE"

        def transcribe_from_bytes(self, audio_bytes: bytes, language=None):
            return "ok"

    class _Owner:
        def __init__(self):
            self.config = {"voice_manager_instance": _VM()}

    class _Meta:
        def __init__(self, artifact_id: str):
            self.artifact_id = artifact_id

    class _Store:
        def __init__(self):
            self._blobs = {}

        def store(self, content: bytes, *, content_type="application/octet-stream", run_id=None, tags=None, artifact_id=None):
            aid = artifact_id or "a1"
            self._blobs[aid] = bytes(content)
            return _Meta(aid)

        def load(self, artifact_id: str):
            b = self._blobs.get(str(artifact_id))
            if b is None:
                return None

            class _Artifact:
                def __init__(self, content: bytes):
                    self.content = content

            return _Artifact(b)

    owner = _Owner()
    cap = _VoiceCapability(owner)

    # bytes mode
    out = cap.tts("hi")
    assert out.startswith(b"RIFF")

    # artifact mode
    store = _Store()
    ref = cap.tts("hi", artifact_store=store)
    assert isinstance(ref, dict)
    assert ref.get("$artifact") == "a1"

    # stt
    assert cap.stt(b"audio") == "ok"


def test_audio_capability_injection_transcribe():
    class _VM:
        def transcribe_from_bytes(self, audio_bytes: bytes, language=None):
            return "ok"

    class _Owner:
        def __init__(self):
            self.config = {"voice_manager_instance": _VM()}

    cap = _AudioCapability(_Owner())
    assert cap.transcribe(b"audio") == "ok"


def test_audio_capability_discovery_surface_supports_generic_contract(monkeypatch):
    import abstractvoice.integrations.abstractcore_plugin as plugin

    _clear_plugin_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://remote.test/v1")
    monkeypatch.setattr(
        plugin,
        "_local_stt_engine_available",
        lambda engine: plugin._norm_engine_id(engine) in {"faster-whisper", "transformers-asr"},
    )

    class _Owner:
        config = {}

    cap = _AudioCapability(_Owner())

    providers = cap.available_providers("audio")
    assert providers["stt"] == ["openai", "openai-compatible", "faster-whisper", "transformers-asr"]
    assert providers["providers"] == providers["stt"]
    assert providers["active_stt_provider"] == "openai"
    assert providers["details"]["stt"]["openai"]["remote"] is True

    assert cap.list_models("audio", provider_id="openai")[:2] == ["gpt-4o-transcribe", "gpt-4o-mini-transcribe"]
    assert cap.list_models("audio", provider="openai:whisper-1") == ["whisper-1"]
    assert "large-v3" in cap.list_models("audio", provider_id="faster-whisper")

    assert cap.available_providers("tts")["providers"] == []
    assert cap.list_models("tts") == []


def test_voice_capability_clone_residency_round_trip(monkeypatch):
    import abstractvoice.integrations.abstractcore_plugin as plugin

    _clear_plugin_env(monkeypatch)
    plugin._VM_CACHE.clear()
    created: list[dict[str, object]] = []

    class _VM:
        def __init__(self, **kwargs):
            created.append(dict(kwargs))
            self.cloning_engine = kwargs.get("cloning_engine", "omnivoice")
            self._loaded: set[str] = set()

        def get_cloned_voice(self, voice_id: str):
            if voice_id == "clone-1":
                return {"voice_id": "clone-1", "engine": "audiodit"}
            return None

        def preload_cloning_engine(self, *, engine=None, voice=None, language=None, speed=None):
            _ = (language, speed)
            eng = str(engine or self.cloning_engine)
            cached_before = eng in self._loaded
            self._loaded.add(eng)
            return {
                "engine": eng,
                "voice_id": voice,
                "state": "resident",
                "resident": True,
                "local": True,
                "unloadable": True,
                "engine_cached": True,
                "engine_cached_before": bool(cached_before),
                "engine_cached_after": True,
                "warmed_via": "engine_preload",
                "runtime_info": {"requested_device": "cpu"},
            }

        def list_resident_components(self):
            return [
                {
                    "component": "cloning_engine",
                    "engine": eng,
                    "state": "resident",
                    "resident": True,
                    "local": True,
                    "unloadable": True,
                    "engine_cached": True,
                    "runtime_info": {"requested_device": "cpu"},
                }
                for eng in sorted(self._loaded)
            ]

        def unload_cloning_engine(self, *, engine=None):
            eng = str(engine or self.cloning_engine)
            unloaded = eng in self._loaded
            self._loaded.discard(eng)
            return {
                "engine": eng,
                "unloaded": unloaded,
                "state": "unloaded" if unloaded else "not_loaded",
                "resident": False,
                "local": True,
                "unloadable": True,
            }

    monkeypatch.setattr("abstractvoice.voice_manager.VoiceManager", _VM)

    class _Owner:
        config = {}

    cap = _VoiceCapability(_Owner())

    loaded = cap.load_resident_model({"task": "tts", "provider": "cloned", "model": "omnivoice"})
    assert loaded["provider"] == "cloned"
    assert loaded["model"] == "omnivoice"
    assert loaded["state"] == "resident"
    assert loaded["loaded"] is True
    assert loaded["resident"] is True
    assert loaded["details"]["warmed_via"] == "engine_preload"
    assert loaded["details"]["engine_cached_before"] is False
    assert loaded["details"]["engine_cached_after"] is True
    assert any(item["cloning_engine"] == "omnivoice" for item in created)

    loaded_repeat = cap.load_resident_model({"task": "tts", "provider": "cloned", "model": "omnivoice"})
    assert loaded_repeat["model"] == "omnivoice"
    assert loaded_repeat["loaded"] is True
    assert loaded_repeat["resident"] is True
    assert loaded_repeat["details"]["engine_cached_before"] is True
    assert loaded_repeat["details"]["engine_cached_after"] is True

    loaded_from_voice = cap.load_resident_model({"task": "tts", "provider": "cloned", "options": {"voice": "clone-1"}})
    assert loaded_from_voice["model"] == "audiodit"
    assert loaded_from_voice["loaded"] is True
    assert loaded_from_voice["options"] == {"voice": "clone-1"}
    assert any(item["cloning_engine"] == "audiodit" for item in created)

    listed = cap.list_resident_models({"task": "tts", "provider": "cloned"})
    assert {(item["provider"], item["model"], item["state"]) for item in listed} >= {
        ("cloned", "audiodit", "resident"),
        ("cloned", "omnivoice", "resident"),
    }

    unloaded = cap.unload_resident_model({"task": "tts", "provider": "cloned", "model": "omnivoice"})
    assert unloaded["model"] == "omnivoice"
    assert unloaded["state"] == "unloaded"
    assert unloaded["loaded"] is False
    assert unloaded["resident"] is False
    assert unloaded["unloaded"] is True
    assert unloaded["details"]["unloaded_count"] >= 1


def test_voice_capability_clone_residency_defers_base_tts_and_stt(monkeypatch):
    _clear_plugin_env(monkeypatch)

    class _Owner:
        config = {"voice_manager_instance": object()}

    cap = _VoiceCapability(_Owner())

    base_tts = cap.load_resident_model({"task": "tts", "provider": "omnivoice", "model": "default"})
    assert base_tts["state"] in {"resident", "configured", "not_implemented"}
    if base_tts["state"] == "not_implemented":
        assert base_tts["loaded"] is False
        assert base_tts["error"]["code"] == "not_implemented_yet"
    else:
        assert base_tts["error"] is None

    stt = cap.load_resident_model({"task": "stt", "provider": "faster-whisper", "model": "base"})
    assert stt["state"] == "not_implemented"
    assert stt["loaded"] is False
    assert stt["error"]["code"] == "not_implemented_yet"
    assert cap.list_resident_models({"task": "stt"}) == []


def test_audio_capability_clone_residency_is_not_implemented(monkeypatch):
    _clear_plugin_env(monkeypatch)

    class _Owner:
        config = {"voice_manager_instance": object()}

    cap = _AudioCapability(_Owner())

    loaded = cap.load_resident_model({"task": "tts", "provider": "cloned", "model": "omnivoice"})
    assert loaded["state"] == "not_implemented"
    assert loaded["loaded"] is False
    assert loaded["error"]["code"] == "not_implemented_yet"
    assert cap.list_resident_models({"task": "tts", "provider": "cloned"}) == []


def test_voice_capability_clone_unload_requires_model_or_voice(monkeypatch):
    _clear_plugin_env(monkeypatch)

    class _Owner:
        config = {"voice_manager_instance": object()}

    cap = _VoiceCapability(_Owner())

    unloaded = cap.unload_resident_model({"task": "tts", "provider": "cloned"})
    assert unloaded["state"] == "failed"
    assert unloaded["error"]["code"] == "invalid_request"


@pytest.mark.basic
def test_audio_capability_prefers_transcribe_file_for_paths_and_artifacts(tmp_path):
    calls = {"file": [], "bytes": 0}

    class _VM:
        def transcribe_file(self, audio_path: str, language=None):
            calls["file"].append((audio_path, language))
            return "ok"

        def transcribe_from_bytes(self, audio_bytes: bytes, language=None):
            calls["bytes"] += 1
            return "nope"

    class _Owner:
        def __init__(self):
            self.config = {"voice_manager_instance": _VM()}

    class _Artifact:
        def __init__(self, content: bytes):
            self.content = content

    class _Store:
        def __init__(self):
            self._blobs = {}

        def store(self, content: bytes, *, content_type="application/octet-stream", run_id=None, tags=None, artifact_id=None):
            aid = artifact_id or "a1"
            self._blobs[aid] = bytes(content)

            class _Meta:
                def __init__(self, artifact_id: str):
                    self.artifact_id = artifact_id

            return _Meta(aid)

        def load(self, artifact_id: str):
            b = self._blobs.get(str(artifact_id))
            if b is None:
                return None
            return _Artifact(b)

    cap = _AudioCapability(_Owner())

    # Path input should call transcribe_file directly.
    p = tmp_path / "clip.webm"
    p.write_bytes(b"WEBM")
    assert cap.transcribe(str(p), language="en") == "ok"
    assert calls["file"] and calls["file"][-1][0] == str(p)

    # Artifact ref input should preserve suffix and still use transcribe_file.
    store = _Store()
    meta = store.store(b"WEBM", content_type="audio/webm")
    ref = {"$artifact": meta.artifact_id, "filename": "clip.webm", "content_type": "audio/webm"}
    assert cap.transcribe(ref, artifact_store=store) == "ok"
    assert calls["bytes"] == 0  # never needed for path/artifact inputs
    assert calls["file"][-1][0].endswith(".webm")
