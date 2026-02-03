import sys
from pathlib import Path

import pytest

# In this workspace, AbstractCore lives in a sibling repo folder (`../abstractcore/`).
REPO_ROOT = Path(__file__).resolve().parents[2]
AF_ROOT = REPO_ROOT.parent
ABSTRACTCORE_REPO = AF_ROOT / "abstractcore"
if ABSTRACTCORE_REPO.exists():
    sys.path.insert(0, str(ABSTRACTCORE_REPO))


@pytest.mark.integration
def test_artifact_tools_file_store_roundtrip(tmp_path: Path):
    try:
        from abstractcore import tool as _tool  # noqa: F401
    except Exception:
        pytest.skip("abstractcore is not importable; skipping voice tool integration tests")

    from abstractvoice.integrations.abstractcore import make_voice_tools
    from abstractvoice.artifacts import RuntimeArtifactStoreAdapter, get_artifact_id

    class FakeVoiceManager:
        def speak_to_bytes(self, text: str, format: str = "wav", voice=None):
            return b"RIFF....WAVE"

        def transcribe_from_bytes(self, audio_bytes: bytes, language=None):
            return "hello world"

    class _Meta:
        def __init__(self, artifact_id: str):
            self.artifact_id = artifact_id

    class FileStore:
        def __init__(self, base: Path):
            self.base = base
            self.base.mkdir(parents=True, exist_ok=True)

        def store(self, content: bytes, *, content_type="application/octet-stream", run_id=None, tags=None, artifact_id=None):
            import hashlib

            b = bytes(content)
            aid = artifact_id or hashlib.sha256(b).hexdigest()[:16]
            (self.base / f"{aid}.bin").write_bytes(b)
            return _Meta(aid)

        def load(self, artifact_id: str):
            p = self.base / f"{artifact_id}.bin"
            if not p.exists():
                return None

            class _Artifact:
                def __init__(self, content: bytes):
                    self.content = content

            return _Artifact(p.read_bytes())

    store1 = FileStore(tmp_path)
    tools = make_voice_tools(voice_manager=FakeVoiceManager(), store=store1)
    by_name = {t._tool_definition.name: t for t in tools if hasattr(t, "_tool_definition")}

    audio_ref = by_name["voice_tts"](text="hi", format="wav")
    aid = get_artifact_id(audio_ref)

    # Restart simulation: new store instance, same base dir.
    store2 = FileStore(tmp_path)
    adapter = RuntimeArtifactStoreAdapter(store2)
    assert adapter.load_bytes(aid).startswith(b"RIFF")

