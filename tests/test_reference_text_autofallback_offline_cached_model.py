from pathlib import Path

import numpy as np
import soundfile as sf

from abstractvoice.cloning.manager import VoiceCloner
from abstractvoice.cloning.store import VoiceCloneStore


def test_reference_text_autofallback_offline_when_model_is_cached(tmp_path: Path, monkeypatch):
    ref = tmp_path / "ref.wav"
    sf.write(str(ref), np.zeros((16000,), dtype=np.float32), 16000, subtype="PCM_16")

    store = VoiceCloneStore(base_dir=tmp_path / "store")
    voice_id = store.create_voice([ref], name="v", reference_text=None, engine="f5_tts")

    class FakeSTT:
        def __init__(self, *args, **kwargs):
            return

        def is_available(self):
            return True

        def transcribe_from_array(self, *_args, **_kwargs):
            return "hello dave"

    # Patch the adapter import used by VoiceCloner to simulate cached offline availability.
    monkeypatch.setattr("abstractvoice.adapters.stt_faster_whisper.FasterWhisperAdapter", FakeSTT)

    cloner = VoiceCloner(store=store, allow_downloads=False)
    txt = cloner._ensure_reference_text(voice_id)
    assert txt.strip().lower().startswith("hello dave")

    # Persisted to store
    info = store.get_voice(voice_id)
    assert (info.reference_text or "").strip()
    assert (info.meta or {}).get("reference_text_source") == "asr"

