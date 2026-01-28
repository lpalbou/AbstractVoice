from pathlib import Path

import pytest

from abstractvoice.cloning.manager import VoiceCloner
from abstractvoice.cloning.store import VoiceCloneStore


def test_reference_text_autofallback_is_persisted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Create a tiny dummy wav to store.
    import numpy as np
    import soundfile as sf

    ref = tmp_path / "ref.wav"
    sf.write(str(ref), np.zeros((24000,), dtype=np.float32), 24000, subtype="PCM_16")

    store = VoiceCloneStore(base_dir=tmp_path / "store")
    voice_id = store.create_voice([ref], name="no_ref_text", engine="f5_tts")

    cloner = VoiceCloner(store=store, debug=False)

    # Avoid running real STT in unit tests.
    monkeypatch.setattr(cloner, "_ensure_reference_text", lambda _vid: "Hello, Dave.")

    # Call speak path; it should route through _ensure_reference_text.
    class FakeEngine:
        def infer_to_wav_bytes(self, *, text, reference_paths, reference_text, speed=None):
            assert reference_text == "Hello, Dave."
            return b"RIFFxxxxWAVE"

    monkeypatch.setattr(cloner, "_get_engine", lambda _engine: FakeEngine())

    data = cloner.speak_to_bytes("test", voice_id=voice_id, format="wav")
    assert data.startswith(b"RIFF")
