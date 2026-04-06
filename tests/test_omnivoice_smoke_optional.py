import os
from pathlib import Path

import pytest


@pytest.mark.integration
def test_omnivoice_tts_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    if os.environ.get("ABSTRACTVOICE_RUN_OMNIVOICE_TESTS") != "1":
        pytest.skip("Set ABSTRACTVOICE_RUN_OMNIVOICE_TESTS=1 to run OmniVoice smoke tests.")

    pytest.importorskip("torch")
    pytest.importorskip("torchaudio")
    pytest.importorskip("transformers")
    pytest.importorskip("omnivoice")

    monkeypatch.setenv("ABSTRACTVOICE_TORCH_DEVICE", "cpu")

    from abstractvoice.omnivoice.runtime import OmniVoiceRuntime, OmniVoiceSettings

    rt = OmniVoiceRuntime(allow_downloads=True, debug=True, device="auto")
    audio, sr = rt.generate_audio(
        text="Bonjour le monde.",
        language="fr",
        instruct=None,
        duration=None,
        speed=1.0,
        settings=OmniVoiceSettings(num_step=2, guidance_scale=0.0, position_temperature=0.0),
    )

    assert int(sr) == 24000
    assert audio is not None
    assert len(audio) > 0


@pytest.mark.integration
def test_omnivoice_cloning_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    if os.environ.get("ABSTRACTVOICE_RUN_OMNIVOICE_TESTS") != "1":
        pytest.skip("Set ABSTRACTVOICE_RUN_OMNIVOICE_TESTS=1 to run OmniVoice smoke tests.")

    pytest.importorskip("torch")
    pytest.importorskip("torchaudio")
    pytest.importorskip("transformers")
    pytest.importorskip("omnivoice")

    monkeypatch.setenv("ABSTRACTVOICE_TORCH_DEVICE", "cpu")

    import numpy as np
    import soundfile as sf

    # Create a tiny prompt wav at 24k.
    sr = 24000
    t = np.arange(int(1.0 * sr), dtype=np.float32) / float(sr)
    prompt = (0.05 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    prompt_path = tmp_path / "prompt.wav"
    sf.write(str(prompt_path), prompt, sr, subtype="PCM_16")

    from abstractvoice.cloning.engine_omnivoice import OmniVoiceVoiceCloningEngine

    eng = OmniVoiceVoiceCloningEngine(
        debug=True,
        device="auto",
        allow_downloads=True,
        num_step=2,
        guidance_scale=0.0,
    )
    wav = eng.infer_to_wav_bytes(
        text="Ceci est un test.",
        reference_paths=[str(prompt_path)],
        reference_text="Bonjour le monde.",
        language="fr",
    )
    assert isinstance(wav, (bytes, bytearray))
    assert wav[:4] == b"RIFF"

