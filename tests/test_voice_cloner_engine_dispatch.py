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


def test_omnivoice_clone_batching_preserves_sentence_boundaries():
    from abstractvoice.cloning.engine_omnivoice import _split_text_batches

    text = (
        "The bridge hums with the low thrum of a starship, but today it feels like an electric guitar solo waiting to be played. "
        "You look up at the viewport and see nothing but swirling void, yet something feels wrong beneath the surface. "
        "Have you ever felt that tiny glitch in your gut whenyou're about to launch into the unknown?"
    )

    chunks = _split_text_batches(text, max_chars=240)

    assert chunks == [
        "The bridge hums with the low thrum of a starship, but today it feels like an electric guitar solo waiting to be played.",
        "You look up at the viewport and see nothing but swirling void, yet something feels wrong beneath the surface.",
        "Have you ever felt that tiny glitch in your gut whenyou're about to launch into the unknown?",
    ]


def test_voice_cloner_rejects_unsupported_reference_file(tmp_path: Path):
    from abstractvoice.cloning.manager import VoiceCloner
    from abstractvoice.cloning.store import VoiceCloneStore

    ref = tmp_path / "ref.mp3"
    ref.write_bytes(b"not audio")

    store = VoiceCloneStore(base_dir=tmp_path / "store")
    cloner = VoiceCloner(store=store, allow_downloads=False)

    with pytest.raises(ValueError, match=r"Unsupported reference audio format"):
        cloner.clone_voice(str(ref), name="v", reference_text="hello.", engine="chroma")


def test_voice_cloner_preload_and_list_loaded_engines(tmp_path: Path):
    from abstractvoice.cloning.manager import VoiceCloner
    from abstractvoice.cloning.store import VoiceCloneStore

    calls: list[str] = []

    class DummyEngine:
        def preload(self):
            calls.append("preload")

        def runtime_info(self):
            return {"requested_device": "cpu"}

        def unload(self):
            calls.append("unload")

    store = VoiceCloneStore(base_dir=tmp_path / "store")
    cloner = VoiceCloner(store=store, allow_downloads=False)
    cloner._engines["omnivoice"] = DummyEngine()

    warmed = cloner.preload_engine("omnivoice")
    assert warmed["engine"] == "omnivoice"
    assert warmed["state"] == "resident"
    assert warmed["resident"] is True
    assert warmed["warmed_via"] == "engine_preload"
    assert warmed["runtime_info"]["requested_device"] == "cpu"

    loaded = cloner.list_loaded_engines()
    assert loaded == [
        {
            "component": "cloning_engine",
            "engine": "omnivoice",
            "state": "resident",
            "resident": True,
            "local": True,
            "unloadable": True,
            "engine_cached": True,
            "runtime_info": {"requested_device": "cpu"},
        }
    ]

    assert cloner.unload_engine("omnivoice") is True
    assert calls == ["preload", "unload"]


def test_voice_cloner_preload_reports_engine_cached_before_and_after(tmp_path: Path, monkeypatch):
    from abstractvoice.cloning.manager import VoiceCloner
    from abstractvoice.cloning.store import VoiceCloneStore

    class DummyEngine:
        def preload(self):
            return None

    store = VoiceCloneStore(base_dir=tmp_path / "store")
    cloner = VoiceCloner(store=store, allow_downloads=False)

    def fake_get_engine(engine: str):
        if engine not in cloner._engines:
            cloner._engines[engine] = DummyEngine()
        return cloner._engines[engine]

    monkeypatch.setattr(cloner, "_get_engine", fake_get_engine)

    warmed = cloner.preload_engine("omnivoice")
    assert warmed["engine_cached_before"] is False
    assert warmed["engine_cached_after"] is True
    assert warmed["engine_cached"] is True

    warmed_repeat = cloner.preload_engine("omnivoice")
    assert warmed_repeat["engine_cached_before"] is True
    assert warmed_repeat["engine_cached_after"] is True
    assert warmed_repeat["engine_cached"] is True


def test_voice_cloner_preload_with_voice_warms_voice_path(tmp_path: Path):
    import numpy as np
    import soundfile as sf

    from abstractvoice.cloning.manager import VoiceCloner
    from abstractvoice.cloning.store import VoiceCloneStore

    ref = tmp_path / "ref.wav"
    sf.write(str(ref), np.zeros((24000,), dtype=np.float32), 24000, subtype="PCM_16")

    store = VoiceCloneStore(base_dir=tmp_path / "store")
    cloner = VoiceCloner(store=store, allow_downloads=False)
    voice_id = cloner.clone_voice(str(ref), name="v", reference_text="hello.", engine="omnivoice")

    calls: list[str] = []

    class DummyEngine:
        def preload(self):
            calls.append("preload")

        def runtime_info(self):
            return {"requested_device": "cpu"}

        def infer_to_wav_bytes(self, *, text, reference_paths, reference_text, speed=None, language=None):
            _ = (reference_paths, reference_text, speed, language)
            calls.append(f"speak:{text}")
            return b"RIFF....dummy"

    cloner._engines["omnivoice"] = DummyEngine()

    warmed = cloner.preload_engine("omnivoice", voice_id=voice_id, language="en")
    assert warmed["voice_id"] == voice_id
    assert warmed["voice_prepared"] is True
    assert warmed["voice_warmed"] is True
    assert warmed["warmed_via"] == "engine_preload+voice_synthesis"
    assert calls == ["preload", "speak:Hello."]
