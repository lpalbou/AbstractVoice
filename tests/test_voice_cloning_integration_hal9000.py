import os
import shutil
from pathlib import Path

import pytest

from abstractvoice import VoiceManager


@pytest.mark.skipif(
    os.environ.get("ABSTRACTVOICE_RUN_CLONING_TESTS") != "1",
    reason="Set ABSTRACTVOICE_RUN_CLONING_TESTS=1 to enable heavy cloning integration tests",
)
def test_hal9000_cloning_synthesizes_wav_bytes():
    if shutil.which("f5-tts_infer-cli") is None:
        pytest.skip("f5-tts_infer-cli not available (install abstractvoice[cloning])")

    ref = Path("audio_samples/hal9000/hal9000_hello.wav")
    if not ref.exists():
        pytest.skip("HAL9000 reference sample not present")

    vm = VoiceManager()
    voice_id = vm.clone_voice(str(ref), name="hal9000_test")

    data = vm.speak_to_bytes("Good evening, Dave.", voice=voice_id, format="wav")
    assert isinstance(data, (bytes, bytearray))
    assert data[:4] == b"RIFF"

