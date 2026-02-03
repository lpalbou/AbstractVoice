import pytest

from abstractvoice.integrations.abstractcore_plugin import _AudioCapability, _VoiceCapability, register


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
