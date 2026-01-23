from pathlib import Path

import numpy as np
import soundfile as sf

from abstractvoice.cloning.store import VoiceCloneStore


def test_voice_clone_store_rename_and_delete(tmp_path: Path):
    ref = tmp_path / "ref.wav"
    sf.write(str(ref), np.zeros((24000,), dtype=np.float32), 24000, subtype="PCM_16")

    store = VoiceCloneStore(base_dir=tmp_path / "store")
    voice_id = store.create_voice([ref], name="old_name", engine="f5_tts")

    store.rename_voice(voice_id, "new_name")
    assert store.get_voice(voice_id).name == "new_name"

    store.delete_voice(voice_id)
    voices = store.list_voices()
    assert not any(v.get("voice_id") == voice_id for v in voices)

