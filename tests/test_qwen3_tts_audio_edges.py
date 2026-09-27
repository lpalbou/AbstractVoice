"""All Qwen synthesis modes de-click utterance edges before any transport."""

from types import SimpleNamespace

import numpy as np
import pytest

from abstractvoice.qwen3_tts.runtime import Qwen3TTSRuntime


@pytest.mark.parametrize("method,kwargs", [
    ("synthesize_custom_voice", {"speaker": "aiden"}),
    ("synthesize_voice_design", {"instruct": "A warm narrator"}),
    ("synthesize_clone", {"clone_prompt": object()}),
])
@pytest.mark.parametrize("sr", [16000, 24000, 48000])
def test_complete_utterance_edges_fade_without_mutating_model_output(method, kwargs, sr):
    raw = np.full(sr // 10, 0.4, dtype=np.float32)
    def generate(**kwargs):
        return [raw], sr
    runtime = Qwen3TTSRuntime(allow_downloads=False)
    runtime._model = SimpleNamespace(
        generate_custom_voice=generate, generate_voice_design=generate,
        generate_voice_clone=generate,
    )
    for _ in range(2):
        audio, sample_rate = getattr(runtime, method)("Hello there.", **kwargs)
        assert sample_rate == sr and audio.shape == raw.shape
        assert audio.dtype == np.float32
        assert audio[0] == 0 and audio[-1] == 0
        n = round(sr * 0.005)
        np.testing.assert_array_equal(audio[n:-n], raw[n:-n])
        assert np.all(np.diff(audio[:n]) >= 0)
        assert np.all(np.diff(audio[-n:]) <= 0)
        np.testing.assert_array_equal(raw, np.full_like(raw, 0.4))
