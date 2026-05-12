import pytest

from abstractvoice.integrations.abstractcore_plugin import _AudioCapability, _VoiceCapability, register


_PLUGIN_ENV_KEYS = (
    "ABSTRACTVOICE_LANGUAGE",
    "ABSTRACTVOICE_ALLOW_DOWNLOADS",
    "ABSTRACTVOICE_TTS_ENGINE",
    "ABSTRACTVOICE_STT_ENGINE",
    "ABSTRACTVOICE_TTS_MODEL",
    "ABSTRACTVOICE_STT_MODEL",
    "ABSTRACTVOICE_REMOTE_BASE_URL",
    "ABSTRACTVOICE_REMOTE_API_KEY",
    "ABSTRACTVOICE_REMOTE_TIMEOUT_S",
    "ABSTRACTVOICE_OPENAI_TTS_MODEL",
    "ABSTRACTVOICE_OPENAI_STT_MODEL",
    "ABSTRACTVOICE_OPENAI_BASE_URL",
    "ABSTRACTVOICE_OPENAI_API_KEY",
    "ABSTRACTVOICE_OPENAI_TIMEOUT_S",
    "ABSTRACTVOICE_OPENAI_COMPATIBLE_TTS_MODEL",
    "ABSTRACTVOICE_OPENAI_COMPATIBLE_STT_MODEL",
    "ABSTRACTVOICE_OPENAI_COMPATIBLE_BASE_URL",
    "ABSTRACTVOICE_OPENAI_COMPATIBLE_API_KEY",
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
    assert seen["remote_api_key"] == "sk-test"


def test_voice_capability_env_overrides_openai_defaults(monkeypatch):
    import abstractvoice.integrations.abstractcore_plugin as plugin

    _clear_plugin_env(monkeypatch)
    monkeypatch.setenv("ABSTRACTVOICE_TTS_ENGINE", "openai-compatible")
    monkeypatch.setenv("ABSTRACTVOICE_STT_ENGINE", "openai-compatible")
    monkeypatch.setenv("ABSTRACTVOICE_OPENAI_COMPATIBLE_BASE_URL", "http://remote.test/v1")
    monkeypatch.setenv("ABSTRACTVOICE_OPENAI_COMPATIBLE_API_KEY", "remote-key")
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
    assert seen["remote_api_key"] == "remote-key"
    assert seen["allow_downloads"] is False
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


def test_voice_capability_catalog_surface_serializes_profiles_and_models():
    from abstractvoice.voice_profiles import VoiceProfile

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
    assert cap.list_stt_models() == ["gpt-active-stt", "gpt-4o-transcribe", "gpt-4o-mini-transcribe", "whisper-1"]

    catalog = cap.voice_catalog()
    assert catalog["engine_id"] == "openai"
    assert catalog["active_profile"]["profile_id"] == "alloy"
    assert catalog["active_model"] == "gpt-active-tts"
    assert catalog["tts_models"] == ["gpt-active-tts", "tts-1", "tts-1-hd", "en_US-lessac-medium.onnx"]
    assert catalog["stt_models"] == ["gpt-active-stt", "gpt-4o-transcribe", "gpt-4o-mini-transcribe", "whisper-1"]
    assert catalog["cloned_voices"][0]["voice_id"] == "clone_laurent"
    assert catalog["voices"][-1]["kind"] == "clone"
    assert catalog["catalog"]["openai"]["alloy"]["remote"] is True


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
