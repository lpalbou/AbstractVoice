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
        def infer_to_wav_bytes(self, *, text, reference_paths, reference_text, speed=None, language=None):
            assert text == "hi"
            assert reference_text == "hello."
            assert len(list(reference_paths)) == 1
            return b"RIFF....dummy"

        def infer_to_audio_chunks(self, *, text, reference_paths, reference_text, speed=None, max_chars=120, language=None):
            yield np.zeros((10,), dtype=np.float32), 24000

    cloner._engines["chroma"] = DummyEngine()

    data = cloner.speak_to_bytes("hi", voice_id=voice_id, format="wav")
    assert data[:4] == b"RIFF"

    chunks = list(cloner.speak_to_audio_chunks("hi", voice_id=voice_id, max_chars=50))
    assert len(chunks) == 1
    audio, sr = chunks[0]
    assert sr == 24000
    assert len(audio) == 10


def test_voice_cloner_clone_from_wav_bytes_sets_engine(tmp_path: Path):
    import io
    import wave
    import numpy as np

    from abstractvoice.cloning.manager import VoiceCloner
    from abstractvoice.cloning.store import VoiceCloneStore

    sr = 24000
    pcm = np.zeros((int(sr * 0.5),), dtype=np.int16).tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)
    wav_bytes = buf.getvalue()

    store = VoiceCloneStore(base_dir=tmp_path / "store")
    cloner = VoiceCloner(store=store, allow_downloads=False)

    voice_id = cloner.clone_voice_from_wav_bytes(wav_bytes, name="v", reference_text="hello.", engine="chroma")
    voice = store.get_voice(voice_id)
    assert voice.engine == "chroma"


def test_voice_cloner_defaults_new_clones_to_omnivoice(tmp_path: Path):
    import numpy as np
    import soundfile as sf

    from abstractvoice.cloning.manager import VoiceCloner
    from abstractvoice.cloning.store import VoiceCloneStore

    ref = tmp_path / "ref.wav"
    sf.write(str(ref), np.zeros((24000,), dtype=np.float32), 24000, subtype="PCM_16")

    class DummyEngine:
        def prepare_voice(self, reference_paths, *, reference_text=None, voice_dir=None, name=None):
            return None

    store = VoiceCloneStore(base_dir=tmp_path / "store")
    cloner = VoiceCloner(store=store, allow_downloads=False)
    cloner._engines["omnivoice"] = DummyEngine()

    voice_id = cloner.clone_voice(str(ref), name="v", reference_text="hello.")
    assert store.get_voice(voice_id).engine == "omnivoice"


def test_omnivoice_clone_high_quality_uses_stable_step_count():
    from abstractvoice.cloning.engine_omnivoice import OmniVoiceVoiceCloningEngine

    engine = OmniVoiceVoiceCloningEngine(allow_downloads=False)
    engine.set_quality_preset("high")
    info = engine.runtime_info()

    assert info["quality_preset"] == "high"
    assert info["num_step"] == 16
    assert info["guidance_scale"] == 2.0


def test_voice_cloner_rejects_unsupported_reference_file(tmp_path: Path):
    from abstractvoice.cloning.manager import VoiceCloner
    from abstractvoice.cloning.store import VoiceCloneStore

    ref = tmp_path / "ref.mp3"
    ref.write_bytes(b"not audio")

    store = VoiceCloneStore(base_dir=tmp_path / "store")
    cloner = VoiceCloner(store=store, allow_downloads=False)

    with pytest.raises(ValueError, match=r"Unsupported reference audio format"):
        cloner.clone_voice(str(ref), name="v", reference_text="hello.", engine="chroma")
