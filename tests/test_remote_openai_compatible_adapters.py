from __future__ import annotations

import io
import json
import os
import wave
from pathlib import Path
from typing import Any

import pytest


def _live_openai_api_key() -> str | None:
    value = os.environ.get("OPENAI_API_KEY")
    if value and value.strip():
        return value.strip()
    return None


class _FakeResponse:
    def __init__(self, *, content: bytes = b"", status_code: int = 200, content_type: str = "application/json"):
        self.content = bytes(content)
        self.status_code = int(status_code)
        self.headers = {"content-type": content_type}
        self.text = self.content.decode("utf-8", errors="replace")

    def json(self):
        return json.loads(self.content.decode("utf-8"))


class _FakeSession:
    def __init__(self):
        self.requests: list[dict[str, Any]] = []
        self.responses: list[_FakeResponse] = []

    def queue(self, response: _FakeResponse) -> None:
        self.responses.append(response)

    def request(self, method: str, url: str, **kwargs: Any):
        self.requests.append({"method": method, "url": url, **kwargs})
        if not self.responses:
            raise AssertionError(f"No queued fake response for {method} {url}")
        return self.responses.pop(0)


def _wav_bytes() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(b"\x00\x00" * 240)
    return buf.getvalue()


def test_voicemanager_default_is_openai_remote_and_transcribes_with_openai(monkeypatch):
    from abstractvoice import VoiceManager

    session = _FakeSession()
    session.queue(_FakeResponse(content=b'{"text":"default remote transcript"}'))
    monkeypatch.setattr("abstractvoice.adapters.openai_compatible_http.requests.Session", lambda: session)

    vm = VoiceManager(remote_api_key="sk-test", allow_downloads=False)

    assert getattr(vm.tts_adapter, "engine_id", "") == "openai"
    assert vm._tts_engine_name == "openai"

    text = vm.transcribe_from_bytes(_wav_bytes(), language="en")

    assert text == "default remote transcript"
    req = session.requests[0]
    assert req["method"] == "POST"
    assert req["url"] == "https://api.openai.com/v1/audio/transcriptions"
    assert req["headers"]["Authorization"] == "Bearer sk-test"
    assert req["data"]["model"] == "gpt-4o-mini-transcribe"
    assert req["data"]["language"] == "en"


def test_voicemanager_auto_resolves_to_openai_remote(monkeypatch):
    from abstractvoice import VoiceManager

    session = _FakeSession()
    session.queue(_FakeResponse(content=b'{"text":"auto remote transcript"}'))
    monkeypatch.setattr("abstractvoice.adapters.openai_compatible_http.requests.Session", lambda: session)

    vm = VoiceManager(
        tts_engine="auto",
        stt_engine="auto",
        remote_api_key="sk-test",
        allow_downloads=False,
    )

    assert getattr(vm.tts_adapter, "engine_id", "") == "openai"
    assert vm._tts_engine_name == "openai"
    assert vm.transcribe_from_bytes(_wav_bytes(), language="en") == "auto remote transcript"
    assert session.requests[0]["url"] == "https://api.openai.com/v1/audio/transcriptions"


def test_voicemanager_default_requires_openai_key(monkeypatch):
    from abstractvoice import VoiceManager

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OpenAI audio requires OPENAI_API_KEY"):
        VoiceManager(allow_downloads=False)


def test_voicemanager_openai_tts_posts_to_speech_endpoint(monkeypatch):
    from abstractvoice import VoiceManager

    session = _FakeSession()
    session.queue(_FakeResponse(content=b"RIFF....WAVE", content_type="audio/wav"))
    monkeypatch.setattr("abstractvoice.adapters.openai_compatible_http.requests.Session", lambda: session)

    vm = VoiceManager(
        tts_engine="openai",
        tts_model="gpt-test-tts",
        remote_api_key="sk-test",
        allow_downloads=False,
    )

    out = vm.speak_to_bytes("Hello.", format="wav")

    assert out == b"RIFF....WAVE"
    req = session.requests[0]
    assert req["method"] == "POST"
    assert req["url"] == "https://api.openai.com/v1/audio/speech"
    assert req["headers"]["Authorization"] == "Bearer sk-test"
    assert req["json"]["model"] == "gpt-test-tts"
    assert req["json"]["input"] == "Hello."
    assert req["json"]["voice"] == "alloy"
    assert req["json"]["response_format"] == "wav"


def test_openai_tts_profiles_are_builtin_and_drive_voice_field(monkeypatch):
    from abstractvoice import VoiceManager

    session = _FakeSession()
    session.queue(_FakeResponse(content=b'{"error":{"message":"not enabled"}}', status_code=404))
    session.queue(_FakeResponse(content=b"RIFFnovaWAVE", content_type="audio/wav"))
    monkeypatch.setattr("abstractvoice.adapters.openai_compatible_http.requests.Session", lambda: session)

    vm = VoiceManager(
        tts_engine="openai",
        remote_api_key="sk-test",
        allow_downloads=False,
    )

    profiles = vm.get_profiles(kind="tts")
    assert any(p.profile_id == "alloy" for p in profiles)
    assert any(p.profile_id == "nova" for p in profiles)
    assert vm.set_profile("nova", kind="tts") is True

    out = vm.speak_to_bytes("Hello.", format="wav")

    assert out == b"RIFFnovaWAVE"
    assert [r["url"] for r in session.requests] == [
        "https://api.openai.com/v1/audio/voices",
        "https://api.openai.com/v1/audio/speech",
    ]
    req = session.requests[-1]
    assert req["url"] == "https://api.openai.com/v1/audio/speech"
    assert req["json"]["voice"] == "nova"


def test_openai_tts_lists_remote_models_and_builtin_profiles(monkeypatch):
    from abstractvoice import VoiceManager

    session = _FakeSession()
    session.queue(
        _FakeResponse(
            content=b"""
            {
              "data": [
                {"id": "gpt-4o-mini-tts"},
                {"id": "tts-1-hd"},
                {"id": "gpt-4o-mini-transcribe"}
              ]
            }
            """,
            content_type="application/json",
        )
    )
    session.queue(
        _FakeResponse(
            content=b"""
            {
              "data": [
                {"id": "voice_custom", "name": "Custom Voice"}
              ]
            }
            """,
            content_type="application/json",
        )
    )
    monkeypatch.setattr("abstractvoice.adapters.openai_compatible_http.requests.Session", lambda: session)

    vm = VoiceManager(
        tts_engine="openai",
        remote_api_key="sk-test",
        allow_downloads=False,
    )

    catalog = vm.list_available_models()

    assert [r["url"] for r in session.requests] == [
        "https://api.openai.com/v1/models",
        "https://api.openai.com/v1/audio/voices",
    ]
    assert "openai" in catalog
    assert "alloy" in catalog["openai"]
    assert "nova" in catalog["openai"]
    assert "voice_custom" in catalog["openai"]
    assert "gpt-4o-mini-tts" in catalog["openai"]["alloy"]["available_models"]
    assert "tts-1-hd" in catalog["openai"]["alloy"]["available_models"]
    assert "gpt-4o-mini-transcribe" not in catalog["openai"]["alloy"]["available_models"]
    assert catalog["openai"]["alloy"]["remote"] is True
    assert catalog["openai"]["alloy"]["model"] == "gpt-4o-mini-tts"


@pytest.mark.integration
def test_live_openai_default_lists_tts_models_and_voice_profiles(monkeypatch):
    api_key = _live_openai_api_key()
    if not api_key:
        pytest.skip("Set OPENAI_API_KEY to run live OpenAI TTS discovery regression.")
    if api_key.lower() in {"test", "sk-test"} or api_key.lower().startswith("sk-test"):
        pytest.skip("Set a real OPENAI_API_KEY to run live OpenAI TTS discovery regression.")

    from abstractvoice import VoiceManager

    monkeypatch.setenv("OPENAI_API_KEY", api_key)
    for name in (
        "OPENAI_BASE_URL",
        "ABSTRACTVOICE_OPENAI_TTS_MODEL",
        "ABSTRACTVOICE_OPENAI_TTS_VOICE",
        "ABSTRACTVOICE_OPENAI_TTS_MODELS",
        "ABSTRACTVOICE_OPENAI_TTS_VOICES",
        "ABSTRACTVOICE_OPENAI_MODEL_PATHS",
        "ABSTRACTVOICE_OPENAI_TTS_MODEL_PATHS",
        "ABSTRACTVOICE_REMOTE_MODEL_PATHS",
        "ABSTRACTVOICE_REMOTE_TTS_MODEL_PATHS",
        "ABSTRACTVOICE_OPENAI_MODEL_PATH",
        "ABSTRACTVOICE_OPENAI_TTS_MODEL_PATH",
        "ABSTRACTVOICE_REMOTE_MODEL_PATH",
        "ABSTRACTVOICE_REMOTE_TTS_MODEL_PATH",
        "ABSTRACTVOICE_OPENAI_VOICE_PROFILE_PATHS",
        "ABSTRACTVOICE_OPENAI_VOICE_LIST_PATHS",
        "ABSTRACTVOICE_REMOTE_VOICE_PROFILE_PATHS",
        "ABSTRACTVOICE_REMOTE_VOICE_LIST_PATHS",
        "ABSTRACTVOICE_OPENAI_VOICE_PROFILE_PATH",
        "ABSTRACTVOICE_OPENAI_VOICE_LIST_PATH",
        "ABSTRACTVOICE_REMOTE_VOICE_PROFILE_PATH",
        "ABSTRACTVOICE_REMOTE_VOICE_LIST_PATH",
    ):
        monkeypatch.delenv(name, raising=False)

    vm = VoiceManager(tts_engine="openai", allow_downloads=False, remote_timeout_s=30)
    assert getattr(vm.tts_adapter, "engine_id", "") == "openai"
    assert getattr(vm.tts_adapter, "base_url", "") == "https://api.openai.com/v1"

    profiles = vm.get_profiles(kind="tts")
    profile_ids = {p.profile_id for p in profiles}
    assert {"alloy", "nova", "shimmer"}.issubset(profile_ids)

    catalog = vm.list_available_models()
    openai_catalog = catalog.get("openai")
    assert openai_catalog
    assert {"alloy", "nova", "shimmer"}.issubset(openai_catalog)

    alloy = openai_catalog["alloy"]
    assert alloy["remote"] is True
    assert alloy["engine"] == "openai"
    assert alloy["provider"] == "openai"
    assert "gpt-4o-mini-tts" in alloy["available_models"]
    assert any("tts" in model_id for model_id in alloy["available_models"])

    remote_model_ids = getattr(vm.tts_adapter, "_remote_tts_models", [])
    assert remote_model_ids
    assert set(remote_model_ids).issubset(set(alloy["available_models"]))


def test_openai_compatible_profiles_load_from_remote_voice_endpoint(monkeypatch):
    from abstractvoice import VoiceManager

    session = _FakeSession()
    session.queue(
        _FakeResponse(
            content=b"""
            {
              "profiles": [{"profile_id": "narrator", "label": "Narrator"}],
              "cloned_voices": [{"voice_id": "clone_alice", "name": "Alice"}]
            }
            """,
            content_type="application/json",
        )
    )
    session.queue(_FakeResponse(content=b"RIFFprofileWAVE", content_type="audio/wav"))
    monkeypatch.setattr("abstractvoice.adapters.openai_compatible_http.requests.Session", lambda: session)

    vm = VoiceManager(
        tts_engine="openai-compatible",
        tts_model="remote-tts",
        remote_base_url="http://remote.test/v1",
        remote_api_key="sk-remote",
        allow_downloads=False,
    )

    profiles = vm.get_profiles(kind="tts")

    assert [r["url"] for r in session.requests] == ["http://remote.test/v1/audio/voices"]
    assert {p.profile_id for p in profiles} >= {"narrator", "clone_alice"}
    assert vm.set_profile("narrator", kind="tts") is True

    out = vm.speak_to_bytes("Read this.", format="wav")

    assert out == b"RIFFprofileWAVE"
    speech_req = session.requests[1]
    assert speech_req["url"] == "http://remote.test/v1/audio/speech"
    assert speech_req["json"]["voice"] == "narrator"
    assert speech_req["json"]["model"] == "remote-tts"


def test_voicemanager_openai_compatible_stt_posts_to_transcriptions(monkeypatch):
    from abstractvoice import VoiceManager

    session = _FakeSession()
    session.queue(_FakeResponse(content=b'{"text":"remote transcript"}'))
    monkeypatch.setattr("abstractvoice.adapters.openai_compatible_http.requests.Session", lambda: session)

    vm = VoiceManager(
        stt_engine="openai-compatible",
        stt_model="remote-whisper",
        remote_base_url="http://remote.test/v1",
        remote_api_key="sk-remote",
        allow_downloads=False,
    )

    text = vm.transcribe_from_bytes(_wav_bytes(), language="en")

    assert text == "remote transcript"
    req = session.requests[0]
    assert req["method"] == "POST"
    assert req["url"] == "http://remote.test/v1/audio/transcriptions"
    assert req["headers"]["Authorization"] == "Bearer sk-remote"
    assert req["data"]["model"] == "remote-whisper"
    assert req["data"]["language"] == "en"
    assert req["data"]["response_format"] == "json"
    filename, content, content_type = req["files"]["file"]
    assert filename == "audio.wav"
    assert content.startswith(b"RIFF")
    assert content_type == "audio/wav"


def test_remote_cloning_engine_stores_remote_voice_id_and_uses_it_for_speech(tmp_path: Path):
    from abstractvoice.cloning.manager import VoiceCloner
    from abstractvoice.cloning.store import VoiceCloneStore

    session = _FakeSession()
    session.queue(_FakeResponse(content=b'{"voice_id":"rv_123"}'))
    session.queue(_FakeResponse(content=b"RIFFremoteWAVE", content_type="audio/wav"))

    ref = tmp_path / "ref.wav"
    ref.write_bytes(_wav_bytes())

    store = VoiceCloneStore(base_dir=tmp_path / "store")
    cloner = VoiceCloner(
        store=store,
        default_engine="openai-compatible",
        remote_base_url="http://remote.test/v1",
        remote_api_key="sk-remote",
        remote_tts_model="remote-tts",
        remote_session=session,
        allow_downloads=False,
    )

    voice_id = cloner.clone_voice(str(ref), name="demo", reference_text="hello.")
    voice = store.get_voice_dict(voice_id)

    assert voice["engine"] == "openai-compatible"
    assert voice["meta"]["remote_voice_id"] == "rv_123"

    out = cloner.speak_to_bytes("Say this.", voice_id=voice_id, format="wav")

    assert out == b"RIFFremoteWAVE"
    clone_req = session.requests[0]
    assert clone_req["url"] == "http://remote.test/v1/voice/clone"
    assert clone_req["data"]["name"] == "demo"
    assert clone_req["data"]["reference_text"] == "hello."
    assert "file" in clone_req["files"]

    speech_req = session.requests[1]
    assert speech_req["url"] == "http://remote.test/v1/audio/speech"
    assert speech_req["json"]["model"] == "remote-tts"
    assert speech_req["json"]["voice"] == "rv_123"
    assert speech_req["json"]["input"] == "Say this."
