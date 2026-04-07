from pathlib import Path

from abstractvoice.cloning.store import VoiceCloneStore


def test_voice_clone_store_roundtrip(tmp_path: Path):
    # Create a tiny dummy wav to store.
    import numpy as np
    import soundfile as sf

    ref = tmp_path / "ref.wav"
    sf.write(str(ref), np.zeros((24000,), dtype=np.float32), 24000, subtype="PCM_16")

    store = VoiceCloneStore(base_dir=tmp_path / "store")
    voice_id = store.create_voice([ref], name="hal9000_test", engine="f5_tts")

    voices = store.list_voices()
    assert any(v.get("voice_id") == voice_id for v in voices)

    v = store.get_voice(voice_id)
    assert v.name == "hal9000_test"
    assert len(v.reference_files) == 1

    export_path = store.export_voice(voice_id, tmp_path / "bundle.zip")
    assert Path(export_path).exists()

    imported_id = store.import_voice(export_path)
    assert imported_id != voice_id
    imported = store.get_voice(imported_id)
    assert imported.name == "hal9000_test"
    assert len(store.resolve_reference_paths(imported_id)) == 1


def test_voice_clone_store_create_from_wav_bytes(tmp_path: Path):
    import io
    import wave
    import numpy as np

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
    voice_id = store.create_voice_from_wav_bytes(
        wav_bytes,
        name="bytes_voice",
        reference_text="hello.",
        engine="f5_tts",
        meta={"source": "bytes_test"},
    )

    v = store.get_voice(voice_id)
    assert v.name == "bytes_voice"
    assert v.engine == "f5_tts"
    assert (v.reference_text or "").strip() == "hello."
    assert isinstance(v.meta, dict)
    assert v.meta.get("source") == "bytes_test"
    assert v.meta.get("reference_text_source") == "manual"

    refs = store.resolve_reference_paths(voice_id)
    assert len(refs) == 1
    assert refs[0].exists()
    assert refs[0].read_bytes()[:4] == b"RIFF"

