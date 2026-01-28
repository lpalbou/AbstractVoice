import importlib.util
import os
from pathlib import Path

import pytest

from abstractvoice.cloning.manager import VoiceCloner
from abstractvoice.cloning.store import VoiceCloneStore


@pytest.mark.skipif(
    os.environ.get("ABSTRACTVOICE_RUN_CHROMA_TESTS") != "1",
    reason="Set ABSTRACTVOICE_RUN_CHROMA_TESTS=1 to enable heavy Chroma integration tests",
)
def test_chroma_cloning_synthesizes_wav_bytes(tmp_path: Path):
    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("transformers") is None:
        pytest.skip("Chroma runtime not installed (install abstractvoice[chroma])")

    from abstractvoice.cloning.engine_chroma import ChromaVoiceCloningEngine

    engine = ChromaVoiceCloningEngine(debug=True)
    if not engine.are_chroma_artifacts_available():
        pytest.skip("Chroma artifacts not present (run: abstractvoice-prefetch --chroma)")

    ref = Path("audio_samples/hal9000/hal9000_hello.wav")
    if not ref.exists():
        pytest.skip("HAL9000 reference sample not present")

    store = VoiceCloneStore(base_dir=tmp_path / "store")
    cloner = VoiceCloner(store=store, allow_downloads=False)
    voice_id = cloner.clone_voice(str(ref), name="hal9000_chroma", reference_text="Hello.", engine="chroma")

    data = cloner.speak_to_bytes("Good evening, Dave.", voice_id=voice_id, format="wav")
    assert isinstance(data, (bytes, bytearray))
    assert data[:4] == b"RIFF"

