import tempfile
from pathlib import Path

import numpy as np

from abstractvoice.omnivoice.prompt_cache import (
    analyze_prompt_audio_mono,
    load_cached_omnivoice_prompt,
    save_cached_omnivoice_prompt,
)


def test_prompt_cache_roundtrip_and_spec_validation() -> None:
    with tempfile.TemporaryDirectory() as td:
        cache_dir = Path(td) / "cache"
        tokens = np.asarray([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int32)
        spec = {"engine_id": "omnivoice", "profile_id": "female_01", "v": 1}

        save_cached_omnivoice_prompt(
            cache_dir,
            ref_audio_tokens=tokens,
            ref_text="hello",
            ref_rms=0.123,
            prompt_spec=spec,
            extra_meta={"note": "unit-test"},
        )

        loaded = load_cached_omnivoice_prompt(cache_dir, expected_prompt_spec=spec)
        assert loaded is not None
        assert loaded.ref_text == "hello"
        assert abs(float(loaded.ref_rms) - 0.123) < 1e-6
        assert np.asarray(loaded.ref_audio_tokens).shape == (2, 4)

        # Spec mismatch should be treated as a cache miss.
        loaded2 = load_cached_omnivoice_prompt(cache_dir, expected_prompt_spec={"engine_id": "omnivoice", "v": 2})
        assert loaded2 is None


def test_analyze_prompt_audio_mono_is_robust() -> None:
    sr = 24000
    t = np.linspace(0, 0.1, int(sr * 0.1), endpoint=False)
    x = 0.1 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    m = analyze_prompt_audio_mono(x, sr)
    assert m["rms"] > 0
    assert m["peak"] > 0
    assert m["p99_diff"] >= 0
    assert m["hf_ratio_6k"] >= 0

