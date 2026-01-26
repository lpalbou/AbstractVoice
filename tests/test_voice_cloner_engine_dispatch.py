from pathlib import Path

import pytest


def test_voice_cloner_dispatches_by_engine(tmp_path: Path):
    import numpy as np
    import soundfile as sf

    from abstractvoice.cloning.manager import VoiceCloner
    from abstractvoice.cloning.store import VoiceCloneStore

    ref = tmp_path / "ref.wav"
    sf.write(str(ref), np.zeros((24000,), dtype=np.float32), 24000, subtype="PCM_16")

    store = VoiceCloneStore(base_dir=tmp_path / "store")
    cloner = VoiceCloner(store=store, allow_downloads=False)

    voice_id = cloner.clone_voice(str(ref), name="v", reference_text="hello.", engine="chroma")
    voice = store.get_voice(voice_id)
    assert voice.engine == "chroma"

    class DummyEngine:
        def infer_to_wav_bytes(self, *, text, reference_paths, reference_text, speed=None):
            assert text == "hi"
            assert reference_text == "hello."
            assert len(list(reference_paths)) == 1
            return b"RIFF....dummy"

        def infer_to_audio_chunks(self, *, text, reference_paths, reference_text, speed=None, max_chars=120):
            yield np.zeros((10,), dtype=np.float32), 24000

    cloner._engines["chroma"] = DummyEngine()

    data = cloner.speak_to_bytes("hi", voice_id=voice_id, format="wav")
    assert data[:4] == b"RIFF"

    chunks = list(cloner.speak_to_audio_chunks("hi", voice_id=voice_id, max_chars=50))
    assert len(chunks) == 1
    audio, sr = chunks[0]
    assert sr == 24000
    assert len(audio) == 10


def test_voice_cloner_rejects_unsupported_reference_file(tmp_path: Path):
    from abstractvoice.cloning.manager import VoiceCloner
    from abstractvoice.cloning.store import VoiceCloneStore

    ref = tmp_path / "ref.mp3"
    ref.write_bytes(b"not audio")

    store = VoiceCloneStore(base_dir=tmp_path / "store")
    cloner = VoiceCloner(store=store, allow_downloads=False)

    with pytest.raises(ValueError, match=r"Unsupported reference audio format"):
        cloner.clone_voice(str(ref), name="v", reference_text="hello.", engine="chroma")
