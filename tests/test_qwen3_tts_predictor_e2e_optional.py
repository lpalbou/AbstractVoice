"""Offline real-checkpoint parity and timings; opt in with the Qwen test flag.

Run with ``ABSTRACTVOICE_RUN_QWEN3_TTS_TESTS=1 pytest -m model_download -s``.
All five snapshots must already be cached. No performance thresholds in CI:
timings are reported with equal audio/frame counts to avoid misleading RTFs.
"""

import json
import os
import time
from unittest.mock import Mock

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from abstractvoice.qwen3_tts.predictor import generate_codebooks
from abstractvoice.qwen3_tts.runtime import KNOWN_MODEL_IDS, Qwen3TTSRuntime

pytestmark = [
    pytest.mark.model_download,
    pytest.mark.skipif(os.environ.get("ABSTRACTVOICE_RUN_QWEN3_TTS_TESTS") != "1", reason="opt-in real Qwen checkpoints"),
]


def sync():
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def rng_states():
    states = [torch.get_rng_state()]
    if torch.backends.mps.is_available():
        states.append(torch.mps.get_rng_state())
    if torch.cuda.is_available():
        states.extend(torch.cuda.get_rng_state_all())
    return states


@pytest.fixture(scope="module")
def reference_voice():
    runtime = Qwen3TTSRuntime(allow_downloads=False)
    text = "This is a reference voice for the speech generation test."
    try:
        audio, sr = runtime.synthesize_custom_voice(text, speaker="aiden", language="English")
        return (audio, sr), text
    finally:
        runtime.unload()


@pytest.mark.parametrize("model_id", KNOWN_MODEL_IDS)
def test_real_predictor_and_whole_utterance_parity(model_id, reference_voice):
    runtime = Qwen3TTSRuntime(model_id, allow_downloads=False)
    try:
        runtime._ensure_loaded()
        predictor = runtime._model.model.talker.code_predictor
        original_generate = predictor.generate
        captured = []
        def capture(**kwargs):
            captured.append(dict(kwargs))
            return original_generate(**kwargs)
        predictor.generate = capture
        kind = runtime.model_type()
        if kind == "base":
            ref, ref_text = reference_voice
            clone_prompt = runtime.build_clone_prompt(ref_audio=ref, ref_text=ref_text, x_vector_only=False)
            def synthesize(text, language):
                return runtime.synthesize_clone(text, clone_prompt=clone_prompt, language=language)
        elif kind == "voice_design":
            def synthesize(text, language):
                return runtime.synthesize_voice_design(text, instruct="A warm, clear female narrator.", language=language)
        else:
            def synthesize(text, language):
                speaker = "aiden" if language == "English" else "vivian"
                return runtime.synthesize_custom_voice(text, speaker=speaker, language=language)

        # Warm both paths before collecting timings; collect real predictor
        # inputs from early/middle/late frames, not just random embeddings.
        predictor._abstractvoice_predictor = "reference"
        synthesize("Hello there.", "English")
        predictor.generate = original_generate
        for index in sorted({0, len(captured) // 2, len(captured) - 1}):
            kw = {k: v for k, v in captured[index].items() if k not in {"output_hidden_states", "return_dict_in_generate"}}
            for sampled, sampler in [(False, "multinomial"), (True, "multinomial"), (True, "exponential")]:
                kw.update(do_sample=sampled, temperature=0.7 if sampled else 1.0,
                          top_k=7 if sampled else 0, top_p=0.8 if sampled else 1.0)
                logits = []
                hook = predictor.register_forward_hook(lambda m, a, result: logits.append(result.logits[:, -1].clone()))
                try:
                    torch.manual_seed(31415 + index)
                    expected = original_generate(**kw, output_hidden_states=True, return_dict_in_generate=True).sequences
                    expected_logits = list(logits)
                    expected_rng = rng_states()
                    logits.clear()
                    torch.manual_seed(31415 + index)
                    predictor._abstractvoice_predictor = "auto"
                    predictor._abstractvoice_sampler = sampler
                    predictor.generate = Mock(side_effect=AssertionError("unexpected reference fallback"))
                    actual = generate_codebooks(predictor, **kw)
                    assert torch.equal(actual, expected)
                    assert all(torch.equal(a, b) for a, b in zip(rng_states(), expected_rng))
                    assert len(logits) == len(expected_logits) == 15
                    for a, b in zip(logits, expected_logits):
                        torch.testing.assert_close(a, b, atol=0, rtol=0)
                finally:
                    predictor.generate = original_generate
                    hook.remove()

        synthesize("Hello there.", "English")
        for text, language, seed in [
            ("This is Qwen speaking through AbstractVoice.", "English", 31415),
            ("你好，这是语音生成测试。", "Chinese", 23),
        ]:
            expected_audio = None
            measurements = []
            for mode, sampler in [
                ("reference", "multinomial"), ("auto", "multinomial"), ("auto", "exponential"),
                ("auto", "exponential"), ("auto", "multinomial"), ("reference", "multinomial"),
            ]:
                predictor._abstractvoice_predictor = mode
                predictor._abstractvoice_sampler = sampler
                torch.manual_seed(seed)
                sync()
                start = time.perf_counter()
                audio, sr = synthesize(text, language)
                sync()
                elapsed = time.perf_counter() - start
                if expected_audio is None:
                    expected_audio = audio
                np.testing.assert_array_equal(audio, expected_audio)
                assert np.isfinite(audio).all() and sr == 24000
                assert audio[0] == audio[-1] == 0
                assert np.sqrt(np.mean(audio ** 2)) > 1e-5
                measurements.append(dict(mode=mode, sampler=sampler, seconds=elapsed, audio_seconds=len(audio)/sr,
                                         rtf=elapsed/(len(audio)/sr)))
            print("QWEN_BENCHMARK " + json.dumps(dict(model=model_id, language=language,
                  device=str(predictor.device), dtype=str(predictor.dtype), measurements=measurements)), flush=True)
    finally:
        runtime.unload()


def test_mps_sampler_distribution_masking_rng_and_extremes():
    from abstractvoice.qwen3_tts.predictor import _sample_exponential_unchecked, _valid_probabilities
    from transformers.generation.logits_process import TopKLogitsWarper, TopPLogitsWarper
    if not torch.backends.mps.is_available():
        pytest.skip("MPS hardware required")
    for batch, vocab in [(1, 4), (3, 32), (16, 2048)]:
        for seed in [0, 1, 23, 31415]:
            torch.manual_seed(seed)
            scores = torch.randn(batch, vocab, device="mps") / 0.7
            scores[:, 0] = -float("inf")
            scores = TopPLogitsWarper(0.8)(None, TopKLogitsWarper(min(7, vocab))(None, scores))
            probs = scores.softmax(-1)
            torch.manual_seed(seed)
            expected = torch.multinomial(probs, 1)
            expected_rng = rng_states()
            torch.manual_seed(seed)
            actual = _sample_exponential_unchecked(probs)
            assert _valid_probabilities(probs) and torch.equal(actual, expected)
            assert all(torch.equal(a, b) for a, b in zip(rng_states(), expected_rng))
            assert (probs.gather(-1, actual) > 0).all()
    torch.manual_seed(37)
    probs = torch.tensor([0.0, 0.1, 0.3, 0.6], device="mps").expand(100000, -1)
    ids = _sample_exponential_unchecked(probs)
    frequencies = torch.bincount(ids.cpu().flatten(), minlength=4) / len(ids)
    assert _valid_probabilities(probs) and frequencies[0] == 0
    torch.testing.assert_close(frequencies, probs[0].cpu(), atol=0.005, rtol=0)
    scores = torch.tensor([[1e30, -1e30, -float("inf")], [-1e30, 1e30, 1e30]], device="mps")
    ids = _sample_exponential_unchecked(scores.softmax(-1))
    assert _valid_probabilities(scores.softmax(-1)) and ids[0] == 0 and ids[1] != 0
    for bad in [float("nan"), float("inf"), -1.0, 0.0]:
        assert not _valid_probabilities(torch.tensor([[0.0, 1.0], [bad, bad]], device="mps"))
