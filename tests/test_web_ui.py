import pytest


def test_web_ui_status_is_lightweight():
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from abstractvoice.examples.web_ui import create_app

    app = create_app(allow_downloads=False)
    client = TestClient(app)

    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["voice_manager_initialized"] is False
    assert data["allow_downloads"] is False
    assert data["defaults"]["cloning_engine"] == "omnivoice"
    assert "optional_dependencies" in data
    assert "omnivoice" in data["optional_dependencies"]
    assert fastapi is not None


def test_web_ui_page_has_role_voice_controls_and_busy_overlay():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from abstractvoice.examples.web_ui import create_app

    app = create_app(allow_downloads=False)
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert 'id="assistant-voice-choice"' in response.text
    assert 'id="user-voice-choice"' in response.text
    assert 'id="tts-engine-choice"' in response.text
    assert 'id="busy-overlay"' in response.text
    assert 'id="conversation-toggle"' in response.text
    assert 'id="clear-chat"' in response.text
    assert 'id="ask-assistant"' in response.text
    assert 'id="llm-provider"' in response.text
    assert 'id="clone-file"' in response.text
    assert 'id="record-clone"' in response.text
    assert 'id="clone-voice"' in response.text
    assert '<option value="omnivoice" selected>OmniVoice</option>' in response.text
    assert 'className = "message-spinner"' in response.text
    assert 'id="read-next"' not in response.text
    assert 'id="read-all"' not in response.text
    assert "playSingleMessage(index)" in response.text
    assert "readConversation(0)" in response.text
    assert "clearChat" in response.text
    assert "MediaRecorder" in response.text
    assert "wavBlobFromRecording" in response.text
    assert "blob = await synthesize(item.text, aborter.signal, speakerRole);" in response.text
    assert '"Synthesizing speech"' not in response.text


def test_web_ui_openapi_documents_request_bodies_and_audio_responses():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from abstractvoice.examples.web_ui import create_app

    app = create_app(allow_downloads=False)
    client = TestClient(app)

    schema = client.get("/openapi.json").json()
    speech = schema["paths"]["/v1/audio/speech"]["post"]
    assert "requestBody" in speech
    speech_schema = schema["components"]["schemas"]["SpeechRequest"]
    assert "input" in speech_schema["properties"]
    assert "response_format" in speech_schema["properties"]
    assert "audio/wav" in speech["responses"]["200"]["content"]

    for path in (
        "/api/voices/select",
        "/api/tts/engine",
        "/api/voices/clone",
        "/v1/voice/clone",
        "/api/tts",
        "/api/stt/transcriptions",
        "/api/stt/transcribe",
        "/api/chat",
        "/v1/audio/speech",
        "/v1/audio/transcriptions",
    ):
        assert "requestBody" in schema["paths"][path]["post"], path


def test_web_ui_openai_speech_alias_maps_to_voice_manager():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from abstractvoice.examples.web_ui import create_app

    class DummyVoiceManager:
        def __init__(self):
            self.calls = []
            self.speed = 1.0

        def speak_to_bytes(self, text, format="wav", voice=None, sanitize_syntax=True):
            self.calls.append(
                {
                    "text": text,
                    "format": format,
                    "voice": voice,
                    "speed": self.speed,
                    "sanitize_syntax": sanitize_syntax,
                }
            )
            return b"RIFFdummy"

        def pop_last_tts_metrics(self):
            return {"engine": "dummy"}

        def get_speed(self):
            return self.speed

        def set_speed(self, speed):
            self.speed = float(speed)
            return True

        def cleanup(self):
            return True

    dummy = DummyVoiceManager()
    app = create_app(voice_manager_factory=lambda _state: dummy)
    client = TestClient(app)

    response = client.post(
        "/v1/audio/speech",
        json={"input": "Hello.", "voice": "clone_1", "response_format": "wav", "speed": 1.25},
    )

    assert response.status_code == 200
    assert response.content == b"RIFFdummy"
    assert response.headers["content-type"].startswith("audio/wav")
    assert dummy.calls == [
        {
            "text": "Hello.",
            "format": "wav",
            "voice": "clone_1",
            "speed": 1.25,
            "sanitize_syntax": True,
        }
    ]
    assert dummy.speed == 1.0


def test_web_ui_transcription_route_maps_to_voice_manager():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from abstractvoice.examples.web_ui import create_app

    class DummyVoiceManager:
        def __init__(self):
            self.calls = []

        def transcribe_file(self, audio_path, language=None):
            self.calls.append({"path_exists": __import__("pathlib").Path(audio_path).exists(), "language": language})
            return "hello transcript"

        def cleanup(self):
            return True

    dummy = DummyVoiceManager()
    app = create_app(voice_manager_factory=lambda _state: dummy)
    client = TestClient(app)

    response = client.post(
        "/api/stt/transcriptions",
        data={"language": "en"},
        files={"file": ("sample.wav", b"fake wav", "audio/wav")},
    )

    assert response.status_code == 200
    assert response.json() == {"ok": True, "text": "hello transcript"}
    assert dummy.calls == [{"path_exists": True, "language": "en"}]


def test_web_ui_all_declared_routes_smoke(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import abstractvoice.examples.web_ui as web_ui

    class DummyProfile:
        engine_id = "piper"
        profile_id = "default"
        label = "Default"
        description = ""

    class DummyVoiceManager:
        def __init__(self):
            self.speed = 1.0

        def get_language(self):
            return "en"

        def get_speed(self):
            return self.speed

        def set_speed(self, speed):
            self.speed = float(speed)
            return True

        def get_active_profile(self, kind="tts"):
            return None

        def get_profiles(self, kind="tts"):
            return [DummyProfile()]

        def set_tts_engine(self, engine, tts_model=None):
            self.tts_engine = engine
            return engine

        def list_cloned_voices(self):
            return [{"voice_id": "clone_a", "name": "Alice", "engine": "dummy"}]

        def list_available_models(self, language=None):
            return {"en_US-test": {"language": language or "en"}}

        def get_supported_languages(self):
            return ["en"]

        def speak_to_bytes(self, text, format="wav", voice=None, sanitize_syntax=True):
            return b"RIFFdummy"

        def pop_last_tts_metrics(self):
            return {"engine": "dummy"}

        def transcribe_file(self, audio_path, language=None):
            return "hello transcript"

        def clone_voice(self, reference_audio_path, name=None, reference_text=None, engine=None):
            return "clone_web"

        def get_cloned_voice(self, voice_id):
            return {"voice_id": voice_id, "name": "Web Voice", "engine": "f5_tts"}

        def cleanup(self):
            return True

    class DummyProvider:
        name = "dummy"
        base_url = "http://dummy.local"

        def list_models(self):
            return ["dummy-model"]

        def chat(self, *, model, messages, temperature=0.4, max_tokens=1024):
            return {"text": "hello", "usage": {}}

    monkeypatch.setattr(web_ui, "resolve_provider", lambda _name: DummyProvider())

    app = web_ui.create_app(voice_manager_factory=lambda _state: DummyVoiceManager())
    client = TestClient(app)

    assert client.get("/").status_code == 200
    assert client.get("/api/status").status_code == 200
    routes = client.get("/api/routes").json()["routes"]
    assert {r["path"] for r in routes} >= {
        "/api/status",
        "/api/voices",
        "/api/tts/engine",
        "/v1/audio/voices",
        "/api/voices/select",
        "/api/voices/clone",
        "/v1/voice/clone",
        "/api/tts",
        "/api/stt/transcriptions",
        "/api/stt/transcribe",
        "/api/llm/models",
        "/api/chat",
        "/v1/audio/speech",
        "/v1/audio/transcriptions",
    }
    assert client.get("/api/voices").status_code == 200
    assert client.post("/api/tts/engine", json={"engine": "piper"}).status_code == 200
    assert client.get("/v1/audio/voices").status_code == 200
    assert client.get("/api/llm/models?provider=ollama").json()["models"] == ["dummy-model"]
    assert client.post("/api/chat", json={"messages": [{"role": "user", "content": "Hi"}]}).status_code == 200
    assert client.post("/api/voices/select", json={"kind": "base", "role": "assistant"}).status_code == 200
    assert client.post("/api/tts", json={"input": "Hi"}).status_code == 200
    assert client.post("/v1/audio/speech", json={"input": "Hi"}).status_code == 200
    assert client.post(
        "/api/stt/transcriptions",
        files={"file": ("sample.wav", b"RIFFdummy", "audio/wav")},
    ).status_code == 200
    assert client.post(
        "/api/stt/transcribe",
        files={"file": ("sample.wav", b"RIFFdummy", "audio/wav")},
    ).status_code == 200
    assert client.post(
        "/v1/audio/transcriptions",
        files={"file": ("sample.wav", b"RIFFdummy", "audio/wav")},
    ).status_code == 200
    assert client.post(
        "/api/voices/clone",
        data={"name": "Web Voice", "engine": "f5_tts", "reference_text": "Hello."},
        files={"file": ("reference.wav", b"RIFFdummy", "audio/wav")},
    ).status_code == 200
    assert client.post(
        "/v1/voice/clone",
        data={"name": "Web Voice", "engine": "f5_tts", "reference_text": "Hello."},
        files={"file": ("reference.wav", b"RIFFdummy", "audio/wav")},
    ).status_code == 200


def test_web_ui_voice_profile_extension_lists_profiles_and_accepts_voice_as_profile():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from abstractvoice.examples.web_ui import create_app

    class DummyProfile:
        engine_id = "openai-compatible"
        profile_id = "narrator"
        label = "Narrator"
        description = "Remote-style profile"
        params = {"voice": "narrator"}
        tags = {"kind": "profile"}

    class DummyVoiceManager:
        def __init__(self):
            self.calls = []
            self.profile = None

        def get_language(self):
            return "en"

        def get_speed(self):
            return 1.0

        def get_active_profile(self, kind="tts"):
            return None

        def get_profiles(self, kind="tts"):
            return [DummyProfile()]

        def set_profile(self, profile_id, kind="tts"):
            self.profile = profile_id
            return profile_id == "narrator"

        def list_cloned_voices(self):
            return [{"voice_id": "clone_a", "name": "Alice", "engine": "dummy"}]

        def list_available_models(self, language=None):
            return {}

        def get_supported_languages(self):
            return ["en"]

        def speak_to_bytes(self, text, format="wav", voice=None, sanitize_syntax=True):
            self.calls.append({"text": text, "voice": voice, "profile": self.profile})
            return b"RIFFdummy"

        def pop_last_tts_metrics(self):
            return {"profile_id": self.profile}

        def cleanup(self):
            return True

    dummy = DummyVoiceManager()
    app = create_app(voice_manager_factory=lambda _state: dummy)
    client = TestClient(app)

    voices = client.get("/v1/audio/voices")
    assert voices.status_code == 200
    payload = voices.json()
    assert payload["object"] == "list"
    assert {item["id"] for item in payload["data"]} >= {"narrator", "clone_a"}

    speech = client.post("/v1/audio/speech", json={"input": "Hi", "voice": "narrator"})
    assert speech.status_code == 200
    assert dummy.calls[-1] == {"text": "Hi", "voice": None, "profile": "narrator"}


def test_web_ui_role_voice_selection_preloads_and_drives_tts():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from abstractvoice.examples.web_ui import create_app

    class DummyVoiceManager:
        def __init__(self):
            self.calls = []
            self.speed = 1.0

        def list_cloned_voices(self):
            return [{"voice_id": "clone_a", "name": "Alice", "engine": "dummy"}]

        def get_profiles(self, kind="tts"):
            return []

        def get_active_profile(self, kind="tts"):
            return None

        def list_available_models(self, language=None):
            return {}

        def get_supported_languages(self):
            return ["en"]

        def get_language(self):
            return "en"

        def get_speed(self):
            return self.speed

        def set_speed(self, speed):
            self.speed = float(speed)
            return True

        def speak_to_bytes(self, text, format="wav", voice=None, sanitize_syntax=True):
            self.calls.append(
                {
                    "text": text,
                    "format": format,
                    "voice": voice,
                    "sanitize_syntax": sanitize_syntax,
                    "speed": self.speed,
                }
            )
            return b"RIFFdummy"

        def pop_last_tts_metrics(self):
            return {"engine": "dummy"}

        def cleanup(self):
            return True

    dummy = DummyVoiceManager()
    app = create_app(voice_manager_factory=lambda _state: dummy)
    client = TestClient(app)

    select_response = client.post(
        "/api/voices/select",
        json={"role": "assistant", "kind": "clone", "voice": "Alice", "preload": True},
    )
    assert select_response.status_code == 200
    selected = select_response.json()
    assert selected["current"]["role_voices"]["assistant"] == "clone_a"
    assert selected["preload"]["ok"] is True
    assert dummy.calls[0]["voice"] == "clone_a"
    assert dummy.calls[0]["sanitize_syntax"] is False

    response = client.post("/api/tts", json={"input": "Role line.", "role": "assistant"})

    assert response.status_code == 200
    assert dummy.calls[-1]["text"] == "Role line."
    assert dummy.calls[-1]["voice"] == "clone_a"


def test_web_ui_tts_engine_switch_resets_role_voice_and_default_profile():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from abstractvoice.examples.web_ui import create_app

    class DummyProfile:
        engine_id = "supertonic"
        profile_id = "M1"
        label = "Supertonic M1"
        description = ""

    class DummyVoiceManager:
        def __init__(self):
            self.engine = "piper"
            self.switches = []

        def set_tts_engine(self, engine, tts_model=None):
            self.switches.append({"engine": engine, "tts_model": tts_model})
            self.engine = engine
            return engine

        def list_cloned_voices(self):
            return [{"voice_id": "clone_a", "name": "Alice", "engine": "dummy"}]

        def get_profiles(self, kind="tts"):
            return [DummyProfile()]

        def get_active_profile(self, kind="tts"):
            return DummyProfile()

        def list_available_models(self, language=None):
            return {}

        def get_supported_languages(self):
            return ["en"]

        def get_language(self):
            return "en"

        def get_speed(self):
            return 1.0

        def speak_to_bytes(self, text, format="wav", voice=None, sanitize_syntax=True):
            return b"RIFFdummy"

        def pop_last_tts_metrics(self):
            return None

        def cleanup(self):
            return True

    dummy = DummyVoiceManager()
    app = create_app(voice_manager_factory=lambda _state: dummy)
    client = TestClient(app)

    selected = client.post(
        "/api/voices/select",
        json={"role": "assistant", "kind": "clone", "voice": "Alice", "preload": False},
    )
    assert selected.status_code == 200
    assert selected.json()["current"]["role_voices"]["assistant"] == "clone_a"

    switched = client.post("/api/tts/engine", json={"engine": "supertonic"})

    assert switched.status_code == 200
    payload = switched.json()
    assert dummy.switches == [{"engine": "supertonic", "tts_model": None}]
    assert payload["tts_engine"] == "supertonic"
    assert payload["current"]["role_voices"]["assistant"] is None
    assert payload["current"]["profile"]["profile_id"] == "M1"


def test_web_ui_failed_role_voice_preload_keeps_previous_selection():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from abstractvoice.examples.web_ui import create_app

    class DummyVoiceManager:
        def list_cloned_voices(self):
            return [
                {"voice_id": "clone_a", "name": "Alice", "engine": "dummy"},
                {"voice_id": "clone_b", "name": "Broken", "engine": "f5_tts"},
            ]

        def speak_to_bytes(self, text, format="wav", voice=None, sanitize_syntax=True):
            if voice == "clone_b":
                raise RuntimeError("OpenF5 artifacts are not present locally.")
            return b"RIFFdummy"

        def pop_last_tts_metrics(self):
            return None

        def cleanup(self):
            return True

    app = create_app(voice_manager_factory=lambda _state: DummyVoiceManager())
    client = TestClient(app)

    ok = client.post(
        "/api/voices/select",
        json={"role": "user", "kind": "clone", "voice": "Alice", "preload": True},
    )
    assert ok.status_code == 200
    assert ok.json()["current"]["role_voices"]["user"] == "clone_a"

    failed = client.post(
        "/api/voices/select",
        json={"role": "user", "kind": "clone", "voice": "Broken", "preload": True},
    )
    assert failed.status_code == 500
    assert "OpenF5 artifacts" in failed.json()["detail"]
    assert client.get("/api/status").json()["current"]["role_voices"]["user"] == "clone_a"


def test_web_ui_clone_voice_route_uploads_reference_audio():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from abstractvoice.examples.web_ui import create_app

    class DummyVoiceManager:
        def __init__(self):
            self.calls = []
            self.speak_calls = []

        def clone_voice(self, reference_audio_path, name=None, reference_text=None, engine=None):
            path = __import__("pathlib").Path(reference_audio_path)
            self.calls.append(
                {
                    "path_exists": path.exists(),
                    "suffix": path.suffix,
                    "bytes": path.read_bytes(),
                    "name": name,
                    "reference_text": reference_text,
                    "engine": engine,
                }
            )
            return "clone_web"

        def get_cloned_voice(self, voice_id):
            return {"voice_id": voice_id, "name": "Web Voice", "engine": "chroma"}

        def speak_to_bytes(self, text, format="wav", voice=None, sanitize_syntax=True):
            self.speak_calls.append(
                {
                    "text": text,
                    "format": format,
                    "voice": voice,
                    "sanitize_syntax": sanitize_syntax,
                }
            )
            return b"RIFFvalidated"

        def pop_last_tts_metrics(self):
            return {"engine": "chroma"}

        def cleanup(self):
            return True

    dummy = DummyVoiceManager()
    app = create_app(voice_manager_factory=lambda _state: dummy)
    client = TestClient(app)

    response = client.post(
        "/api/voices/clone",
        data={"name": "Web Voice", "engine": "chroma", "reference_text": "Hello reference."},
        files={"file": ("reference.wav", b"RIFFdummy", "audio/wav")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["voice_id"] == "clone_web"
    assert data["name"] == "Web Voice"
    assert data["engine"] == "chroma"
    assert data["validation"]["ok"] is True
    assert dummy.calls == [
        {
            "path_exists": True,
            "suffix": ".wav",
            "bytes": b"RIFFdummy",
            "name": "Web Voice",
            "reference_text": "Hello reference.",
            "engine": "chroma",
        }
    ]
    assert dummy.speak_calls == [
        {
            "text": "Hello.",
            "format": "wav",
            "voice": "clone_web",
            "sanitize_syntax": False,
        }
    ]


def test_web_ui_clone_voice_route_removes_unusable_clone_after_validation_failure():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from abstractvoice.examples.web_ui import create_app

    class DummyVoiceManager:
        def __init__(self):
            self.deleted = []

        def clone_voice(self, reference_audio_path, name=None, reference_text=None, engine=None):
            return "clone_broken"

        def get_cloned_voice(self, voice_id):
            return {"voice_id": voice_id, "name": "Broken", "engine": "omnivoice"}

        def speak_to_bytes(self, text, format="wav", voice=None, sanitize_syntax=True):
            try:
                raise RuntimeError("Load failed")
            except RuntimeError as e:
                raise RuntimeError("Failed to load OmniVoice model: missing local snapshot") from e

        def delete_cloned_voice(self, voice_id):
            self.deleted.append(voice_id)
            return True

        def cleanup(self):
            return True

    dummy = DummyVoiceManager()
    app = create_app(voice_manager_factory=lambda _state: dummy)
    client = TestClient(app)

    response = client.post(
        "/api/voices/clone",
        data={"name": "Broken", "engine": "omnivoice", "reference_text": "Bonjour."},
        files={"file": ("reference.wav", b"RIFFdummy", "audio/wav")},
    )

    assert response.status_code == 500
    detail = response.json()["detail"]
    assert "failed validation" in detail
    assert "Failed to load OmniVoice model" in detail
    assert "Load failed" in detail
    assert dummy.deleted == ["clone_broken"]


def test_web_ui_clone_voice_route_rejects_browser_webm_upload():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from abstractvoice.examples.web_ui import create_app

    class DummyVoiceManager:
        def cleanup(self):
            return True

    app = create_app(voice_manager_factory=lambda _state: DummyVoiceManager())
    client = TestClient(app)

    response = client.post(
        "/api/voices/clone",
        data={"name": "Web Voice", "engine": "f5_tts"},
        files={"file": ("reference.webm", b"webm", "audio/webm")},
    )

    assert response.status_code == 400
    assert "WAV, FLAC, or OGG" in response.json()["detail"]


def test_web_ui_remote_clone_default_accepts_remote_audio_formats():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from abstractvoice.examples.web_ui import create_app

    class DummyVoiceManager:
        def __init__(self):
            self.calls = []

        def clone_voice(self, reference_audio_path, name=None, reference_text=None, engine=None):
            path = __import__("pathlib").Path(reference_audio_path)
            self.calls.append({"suffix": path.suffix, "engine": engine, "bytes": path.read_bytes()})
            return "remote_clone"

        def get_cloned_voice(self, voice_id):
            return {"voice_id": voice_id, "name": "Remote", "engine": "openai-compatible"}

        def cleanup(self):
            return True

    dummy = DummyVoiceManager()
    app = create_app(cloning_engine="openai-compatible", voice_manager_factory=lambda _state: dummy)
    client = TestClient(app)

    response = client.post(
        "/v1/voice/clone",
        data={"name": "Remote", "reference_text": "Hello."},
        files={"file": ("reference.webm", b"webm", "audio/webm")},
    )

    assert response.status_code == 200
    assert response.json()["voice_id"] == "remote_clone"
    assert dummy.calls == [{"suffix": ".webm", "engine": None, "bytes": b"webm"}]


def test_web_ui_openai_alias_ignores_local_role_without_voice_selection():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from abstractvoice.examples.web_ui import create_app

    class DummyVoiceManager:
        def __init__(self):
            self.calls = []

        def speak_to_bytes(self, text, format="wav", voice=None, sanitize_syntax=True):
            self.calls.append({"text": text, "voice": voice, "format": format})
            return b"RIFFdummy"

        def pop_last_tts_metrics(self):
            return None

        def cleanup(self):
            return True

    dummy = DummyVoiceManager()
    app = create_app(voice_manager_factory=lambda _state: dummy)
    client = TestClient(app)

    response = client.post("/v1/audio/speech", json={"input": "Hello.", "role": "assistant"})

    assert response.status_code == 200
    assert dummy.calls == [{"text": "Hello.", "voice": None, "format": "wav"}]


def test_web_ui_chat_route_proxies_openai_compatible_provider(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import abstractvoice.examples.web_ui as web_ui

    class DummyProvider:
        name = "dummy"
        base_url = "http://dummy.local"

        def __init__(self):
            self.calls = []

        def chat(self, *, model, messages, temperature=0.4, max_tokens=1024):
            self.calls.append(
                {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
            )
            return {"text": "Hello from the model.", "usage": {"completion_tokens": 5}}

    provider = DummyProvider()
    monkeypatch.setattr(web_ui, "resolve_provider", lambda _name: provider)

    app = web_ui.create_app(allow_downloads=False)
    client = TestClient(app)

    response = client.post(
        "/api/chat",
        json={
            "provider": "ollama",
            "model": "local-model",
            "system_prompt": "Be brief.",
            "messages": [{"role": "user", "content": "Hi"}],
            "temperature": 9,
            "max_tokens": 999999,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["text"] == "Hello from the model."
    assert data["message"] == {"role": "assistant", "content": "Hello from the model."}
    assert data["usage"] == {"completion_tokens": 5}
    assert provider.calls == [
        {
            "model": "local-model",
            "messages": [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "Hi"},
            ],
            "temperature": 2.0,
            "max_tokens": 32768,
        }
    ]


def test_web_ui_chat_route_requires_user_message():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from abstractvoice.examples.web_ui import create_app

    app = create_app(allow_downloads=False)
    client = TestClient(app)

    response = client.post("/api/chat", json={"messages": [{"role": "assistant", "content": "Hi"}]})

    assert response.status_code == 400
    assert "at least one user message" in response.json()["detail"]
