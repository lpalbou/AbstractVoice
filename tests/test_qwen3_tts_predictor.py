"""Tiny real-model parity and fail-closed generation-contract tests (no weights)."""

from unittest.mock import Mock
import importlib.metadata
import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from packaging.version import Version

if Version(importlib.metadata.version("transformers")).major < 5:
    # The vendored model is written for Transformers 5 (the qwen3-tts extra floors
    # 5.9, Python 3.10+); on the 4.x that Python 3.9's stt-hf resolves it cannot run.
    pytest.skip("Qwen3-TTS needs Transformers 5 (the qwen3-tts extra floors 5.9)", allow_module_level=True)

from abstractvoice.qwen3_tts.configuration_qwen3_tts import Qwen3TTSTalkerConfig
from abstractvoice.qwen3_tts.modeling_qwen3_tts import Qwen3TTSTalkerCodePredictorModelForConditionalGeneration
from abstractvoice.qwen3_tts.predictor import generate_codebooks
from abstractvoice.qwen3_tts.predictor import _sample_exponential_unchecked, _valid_probabilities


@pytest.fixture
def predictor():
    torch.manual_seed(7)
    config = Qwen3TTSTalkerConfig(
        hidden_size=32, num_code_groups=4,
        code_predictor_config=dict(
            vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
            head_dim=8, num_code_groups=4,
        ),
    )
    return Qwen3TTSTalkerCodePredictorModelForConditionalGeneration(
        config.code_predictor_config, config,
    ).eval()


def arguments(predictor, batch=1, **overrides):
    return dict(
        inputs_embeds=torch.randn(batch, 2, 32, dtype=predictor.dtype, device=predictor.device),
        max_new_tokens=3, do_sample=True, temperature=0.7, top_k=7, top_p=0.8,
        **overrides,
    )


def reference(predictor, kwargs):
    return predictor.generate(
        **kwargs, output_hidden_states=True, return_dict_in_generate=True,
    ).sequences


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("sampler", ["multinomial", "exponential"])
@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("settings", [
    dict(do_sample=False, temperature=1.0, top_k=0, top_p=1.0),
    dict(do_sample=True, temperature=0.7, top_k=7, top_p=0.8),
    dict(do_sample=True, temperature=1.0, top_k=0, top_p=1.0),
    dict(do_sample=True, temperature=1.5, top_k=1, top_p=0.2),
])
def test_loop_matches_reference_logits_tokens_and_rng(predictor, dtype, batch, settings, sampler):
    predictor.to(dtype=dtype)
    predictor._abstractvoice_sampler = sampler
    kwargs = arguments(predictor, batch)
    kwargs.update(settings)
    logits = []
    hook = predictor.register_forward_hook(lambda m, a, result: logits.append(result.logits[:, -1].clone()))
    try:
        for seed in [0, 17]:
            torch.manual_seed(seed)
            logits.clear()
            expected = reference(predictor, kwargs)
            expected_logits = list(logits)
            expected_rng = torch.get_rng_state()
            original = predictor.generate
            predictor.generate = Mock(side_effect=AssertionError("unexpected reference fallback"))
            try:
                torch.manual_seed(seed)
                logits.clear()
                actual = generate_codebooks(predictor, **kwargs)
            finally:
                predictor.generate = original
            assert torch.equal(actual, expected)
            assert torch.equal(torch.get_rng_state(), expected_rng)
            assert len(logits) == len(expected_logits) == 3
            for a, b in zip(logits, expected_logits):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
    finally:
        hook.remove()


@pytest.mark.parametrize("key,value", [
    ("forced_eos_token_id", 23), ("eos_token_id", 23), ("repetition_penalty", 1.1),
    ("num_beams", 2), ("suppress_tokens", [3]), ("min_new_tokens", 2),
    ("max_time", 1.0), ("cache_implementation", "static"),
    ("output_scores", True), ("sequence_bias", {(3,): 2.0}),
    ("custom_future_option", True),
    # Fields later Transformers 5.x releases added; neutral only when unset.
    ("use_mtp", True), ("max_cache_len", 64),
    ("assistant_ensemble_weight", 0.5), ("speculation_type", "dflash"),
])
def test_nonstandard_config_falls_back_before_inference(predictor, key, value, monkeypatch):
    kwargs = arguments(predictor)
    # First call succeeds: guards must notice subsequent config mutations.
    generate_codebooks(predictor, **kwargs)
    setattr(predictor.generation_config, key, value)
    expected = torch.tensor([[4, 5, 6]])
    predictor.generate = Mock(return_value=Mock(sequences=expected))
    predictor.forward = Mock(side_effect=AssertionError("must not begin fast inference"))
    # Inspect the call independently of application logging configuration and
    # Transformers' process-global warning_once cache.
    warning = Mock()
    monkeypatch.setattr(generate_codebooks.__wrapped__.__globals__["logger"], "warning_once", warning)
    rng = torch.get_rng_state()
    assert generate_codebooks(predictor, **kwargs) is expected
    assert torch.equal(torch.get_rng_state(), rng)
    predictor.generate.assert_called_once_with(
        **kwargs, output_hidden_states=True, return_dict_in_generate=True,
    )
    warning.assert_called_once()
    assert "#FALLBACK" in warning.call_args.args[0]


def test_installed_transformers_defaults_are_all_classified(predictor):
    """Every resolved GenerationConfig field is known and neutral at its default.

    A Transformers release that adds a field makes the guarded loop fall back
    (safe, but slow); this names the field so it is classified deliberately.
    """
    from abstractvoice.qwen3_tts import predictor as module

    config, _ = predictor._prepare_generation_config(
        None, **arguments(predictor), output_hidden_states=True, return_dict_in_generate=True,
    )
    unclassified = {
        key: value for key, value in config.to_dict().items()
        if key not in module._OVERRIDDEN and value not in module._NEUTRAL.get(key, ())
    }
    assert unclassified == {}


def test_future_library_semantic_default_is_not_implicitly_accepted(predictor, monkeypatch):
    from transformers import GenerationConfig
    old = GenerationConfig._get_default_generation_params
    monkeypatch.setattr(GenerationConfig, "_get_default_generation_params",
                        staticmethod(lambda: dict(old(), min_p=0.1)))
    predictor.generate = Mock(return_value=Mock(sequences=torch.tensor([[1, 2, 3]])))
    generate_codebooks(predictor, **arguments(predictor))
    predictor.generate.assert_called_once()


def test_forced_eos_mutation_retains_actual_reference_semantics(predictor):
    kwargs = arguments(predictor)
    generate_codebooks(predictor, **kwargs)
    predictor.generation_config.forced_eos_token_id = 23
    torch.manual_seed(17)
    expected = reference(predictor, kwargs)
    torch.manual_seed(17)
    actual = generate_codebooks(predictor, **kwargs)
    assert torch.equal(actual, expected) and actual[0, -1] == 23


@pytest.mark.parametrize("mode", ["reference", "training", "prefix", "count", "sliding"])
def test_unsupported_model_contract_uses_reference(predictor, mode):
    kwargs = arguments(predictor)
    if mode == "reference":
        predictor._abstractvoice_predictor = "reference"
    elif mode == "training":
        predictor.train()
    elif mode == "prefix":
        kwargs["inputs_embeds"] = torch.randn(1, 3, 32)
    elif mode == "count":
        kwargs["max_new_tokens"] = 2
    else:
        predictor.config.layer_types = ["sliding_attention"] * 2
    predictor.generate = Mock(return_value=Mock(sequences=torch.tensor([[1, 2, 3]])))
    generate_codebooks(predictor, **kwargs)
    predictor.generate.assert_called_once()


def test_failures_are_not_retried_after_rng_consumption(predictor):
    kwargs = arguments(predictor)
    predictor.generate = Mock(side_effect=AssertionError("must not retry"))
    def fail(*args, **kwargs):
        torch.rand(1)
        raise RuntimeError("model failed")
    predictor.forward = fail
    with pytest.raises(RuntimeError, match="model failed"):
        generate_codebooks(predictor, **kwargs)
    predictor.generate.assert_not_called()


@pytest.mark.parametrize("field,options", [
    ("predictor", "one of auto, reference"),
    ("sampler", "one of multinomial, exponential"),
])
def test_invalid_runtime_mode_is_rejected_before_loading(field, options):
    from abstractvoice.qwen3_tts.runtime import Qwen3TTSRuntime, Qwen3TTSSettings
    with pytest.raises(ValueError, match=options):
        Qwen3TTSSettings(**{field: "typo"})
    # The fields are plain attributes: a value reassigned later is still
    # rejected before any snapshot is resolved.
    settings = Qwen3TTSSettings()
    setattr(settings, field, "typo")
    runtime = Qwen3TTSRuntime(allow_downloads=False, settings=settings)
    runtime.snapshot_dir = Mock(side_effect=AssertionError("must validate before loading"))
    with pytest.raises(ValueError, match=options):
        runtime._ensure_loaded()


@pytest.mark.parametrize("settings,expected", [
    ({}, ("auto", "multinomial")),
    ({"sampler": "exponential"}, ("auto", "exponential")),
    ({"predictor": "reference"}, ("reference", "multinomial")),
])
def test_runtime_settings_select_predictor_and_sampler_at_load(monkeypatch, settings, expected):
    """The switches are runtime settings, never process environment."""
    from types import SimpleNamespace

    from abstractvoice.qwen3_tts import orchestration
    from abstractvoice.qwen3_tts.runtime import Qwen3TTSRuntime, Qwen3TTSSettings

    # A stray variable from the pre-release env-var design must have no effect.
    monkeypatch.setenv("ABSTRACTVOICE_QWEN3_TTS_PREDICTOR", "typo")
    monkeypatch.setenv("ABSTRACTVOICE_QWEN3_TTS_SAMPLER", "typo")
    code_predictor = SimpleNamespace()
    inner = SimpleNamespace(
        talker=SimpleNamespace(code_predictor=code_predictor),
        speech_tokenizer=SimpleNamespace(model=SimpleNamespace(to=lambda **_kw: "codec")),
        device="cpu", tts_model_type="custom_voice", to=lambda _device: None, eval=lambda: None,
    )
    monkeypatch.setattr(
        orchestration.Qwen3TTSModel, "from_pretrained", classmethod(lambda cls, *_a, **_kw: SimpleNamespace(model=inner))
    )
    runtime = Qwen3TTSRuntime(allow_downloads=False, device="cpu", settings=Qwen3TTSSettings(**settings))
    runtime.snapshot_dir = Mock(return_value="/nonexistent-snapshot")
    runtime._ensure_loaded()
    assert (code_predictor._abstractvoice_predictor, code_predictor._abstractvoice_sampler) == expected
    info = runtime.runtime_info()
    assert info["predictor_mode"] == expected[0]
    assert info["sampler"] == ("multinomial" if expected[0] == "reference" else expected[1])


@pytest.mark.parametrize("seed", [0, 1, 31415])
@pytest.mark.parametrize("batch,vocab", [(1, 4), (3, 32), (16, 2048)])
def test_exponential_sampler_matches_stock_tokens_and_rng(seed, batch, vocab):
    probabilities = torch.randn(batch, vocab).softmax(-1)
    probabilities[:, 0] = 0
    torch.manual_seed(seed)
    expected = torch.multinomial(probabilities, 1)
    rng = torch.get_rng_state()
    torch.manual_seed(seed)
    actual = _sample_exponential_unchecked(probabilities)
    assert _valid_probabilities(probabilities) and torch.equal(actual, expected)
    assert torch.equal(torch.get_rng_state(), rng)
    assert (actual != 0).all()


def test_exponential_sampler_distribution_and_zero_noise(monkeypatch):
    torch.manual_seed(37)
    probabilities = torch.tensor([0.0, 0.1, 0.3, 0.6]).expand(100000, -1)
    actual = _sample_exponential_unchecked(probabilities)
    frequency = torch.bincount(actual.flatten(), minlength=4) / len(actual)
    assert _valid_probabilities(probabilities) and frequency[0] == 0
    torch.testing.assert_close(frequency, probabilities[0], atol=0.005, rtol=0)
    # Deterministically exercise a zero draw on a masked token.
    monkeypatch.setattr(torch.Tensor, "exponential_", lambda self: self.zero_())
    actual = _sample_exponential_unchecked(torch.tensor([[0.0, 1.0, 0.0]]))
    assert actual.item() == 1


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0, 0.0])
def test_sampler_rejects_invalid_probabilities(bad):
    assert not _valid_probabilities(torch.tensor([[0.0, 1.0], [bad, bad]]))


@pytest.mark.parametrize("step", [0, 1, 2])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("device", ["cpu", pytest.param("mps", marks=[
    pytest.mark.model_download,
    pytest.mark.skipif(os.environ.get("ABSTRACTVOICE_RUN_QWEN3_TTS_TESTS") != "1", reason="opt-in MPS checks"),
])])
def test_invalid_logits_fail_atomically_at_any_codebook(predictor, step, bad, device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS hardware required")
    predictor.to(device)
    kwargs = arguments(predictor, batch=2)
    predictor._abstractvoice_sampler = "exponential"
    predictor.generate = Mock(side_effect=AssertionError("must never retry after sampling"))
    def corrupt(module, args, result):
        if result.generation_steps == step + 1:
            result.logits[1, -1, :] = bad
    hook = predictor.register_forward_hook(corrupt)
    try:
        with pytest.raises(RuntimeError, match="probability tensor"):
            generate_codebooks(predictor, **kwargs)
    finally:
        hook.remove()
    predictor.generate.assert_not_called()


@pytest.mark.parametrize("sliding_window", [4, 64])
def test_codec_decoder_transformer_runs_on_installed_transformers(sliding_window):
    """The codec's decoder transformer builds its masks through masking_utils.

    Weight-free guard for drift in Transformers' mask helpers (cache_position
    left their signature in 5.9); every real decode goes through this model.
    """
    from abstractvoice.qwen3_tts.configuration_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2DecoderConfig
    from abstractvoice.qwen3_tts.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2DecoderTransformerModel

    torch.manual_seed(3)
    config = Qwen3TTSTokenizerV2DecoderConfig(
        hidden_size=16, latent_dim=8, num_attention_heads=2, num_key_value_heads=2, head_dim=8,
        intermediate_size=32, num_hidden_layers=2, sliding_window=sliding_window,
    )
    model = Qwen3TTSTokenizerV2DecoderTransformerModel(config).eval()
    with torch.no_grad():
        hidden = model(inputs_embeds=torch.randn(1, 10, 8)).last_hidden_state
    assert hidden.shape == (1, 10, 8) and torch.isfinite(hidden).all()
