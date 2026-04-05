import os
from pathlib import Path

import pytest


@pytest.mark.integration
def test_audiodit_tts_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    if os.environ.get("ABSTRACTVOICE_RUN_AUDIODIT_TESTS") != "1":
        pytest.skip("Set ABSTRACTVOICE_RUN_AUDIODIT_TESTS=1 to run AudioDiT smoke tests.")

    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("einops")

    monkeypatch.setenv("ABSTRACTVOICE_TORCH_DEVICE", "cpu")

    from abstractvoice.audiodit.runtime import AudioDiTRuntime, AudioDiTSettings

    rt = AudioDiTRuntime(allow_downloads=True, debug=True)
    audio, sr = rt.generate(
        text="Hello world.",
        language="en",
        settings=AudioDiTSettings(steps=2, cfg_strength=0.0, guidance_method="cfg", seed=123),
    )

    assert int(sr) == 24000
    assert audio is not None
    assert len(audio) > 0


@pytest.mark.integration
def test_audiodit_cloning_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    if os.environ.get("ABSTRACTVOICE_RUN_AUDIODIT_TESTS") != "1":
        pytest.skip("Set ABSTRACTVOICE_RUN_AUDIODIT_TESTS=1 to run AudioDiT smoke tests.")

    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("einops")

    monkeypatch.setenv("ABSTRACTVOICE_TORCH_DEVICE", "cpu")

    import numpy as np
    import soundfile as sf

    # Create a tiny prompt wav at 24k.
    sr = 24000
    t = np.arange(int(0.8 * sr), dtype=np.float32) / float(sr)
    prompt = (0.1 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    prompt_path = tmp_path / "prompt.wav"
    sf.write(str(prompt_path), prompt, sr, subtype="PCM_16")

    from abstractvoice.cloning.engine_audiodit import AudioDiTVoiceCloningEngine

    eng = AudioDiTVoiceCloningEngine(
        debug=True,
        device="auto",
        allow_downloads=True,
        steps=2,
        cfg_strength=0.0,
        guidance_method="cfg",
    )
    wav = eng.infer_to_wav_bytes(
        text="Testing cloning.",
        reference_paths=[str(prompt_path)],
        reference_text="Hello.",
        language="en",
    )
    assert isinstance(wav, (bytes, bytearray))
    assert wav[:4] == b"RIFF"

