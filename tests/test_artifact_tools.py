import sys
import tempfile
import unittest
from pathlib import Path

# In this workspace, AbstractCore lives in a sibling repo folder (`../abstractcore/`).
REPO_ROOT = Path(__file__).resolve().parents[1]
AF_ROOT = REPO_ROOT.parent
ABSTRACTCORE_REPO = AF_ROOT / "abstractcore"
if ABSTRACTCORE_REPO.exists():
    sys.path.insert(0, str(ABSTRACTCORE_REPO))


class TestAbstractCoreArtifactTools(unittest.TestCase):
    def test_make_voice_tools_tts_and_transcribe_roundtrip(self):
        try:
            from abstractcore import tool as _tool  # noqa: F401
        except Exception:
            self.skipTest("abstractcore is not importable; skipping voice tool integration tests")

        from abstractvoice.integrations.abstractcore import make_voice_tools

        class FakeVoiceManager:
            def speak_to_bytes(self, text: str, format: str = "wav", voice=None):
                return b"RIFF....WAVE"

            def transcribe_from_bytes(self, audio_bytes: bytes, language=None):
                return "hello world"

        class _Meta:
            def __init__(self, artifact_id: str):
                self.artifact_id = artifact_id

        class _Store:
            def __init__(self, base: Path):
                self._base = base
                self._base.mkdir(parents=True, exist_ok=True)

            def store(self, content: bytes, *, content_type="application/octet-stream", run_id=None, tags=None, artifact_id=None):
                aid = artifact_id or "a1"
                (self._base / f"{aid}.bin").write_bytes(bytes(content))
                return _Meta(aid)

            def load(self, artifact_id: str):
                p = self._base / f"{artifact_id}.bin"
                if not p.exists():
                    return None

                class _Artifact:
                    def __init__(self, content: bytes):
                        self.content = content

                return _Artifact(p.read_bytes())

        with tempfile.TemporaryDirectory() as td:
            store = _Store(Path(td))
            tools = make_voice_tools(voice_manager=FakeVoiceManager(), store=store)

            by_name = {t._tool_definition.name: t for t in tools if hasattr(t, "_tool_definition")}
            self.assertIn("voice_tts", by_name)
            self.assertIn("audio_transcribe", by_name)

            audio_ref = by_name["voice_tts"](text="hi", format="wav")
            self.assertIsInstance(audio_ref, dict)
            self.assertIn("$artifact", audio_ref)
            self.assertEqual(audio_ref.get("content_type"), "audio/wav")

            out = by_name["audio_transcribe"](audio_artifact=audio_ref)
            self.assertIsInstance(out, dict)
            self.assertEqual(out.get("text"), "hello world")
            self.assertIsInstance(out.get("transcript_artifact"), dict)
            self.assertIn("$artifact", out.get("transcript_artifact"))


if __name__ == "__main__":
    unittest.main()

