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
    # The Transformers floor has its own tests (test_transformers_floor.py); this
    # one is about settings and must not depend on the installed release.
    monkeypatch.setattr("abstractvoice.transformers_floor.require_transformers", lambda *_a, **_kw: None)
    # The switches wire the PyTorch code predictor: pin that backend (ADR 0008 would pick
    # MLX on an Apple silicon machine with `mlx` installed).
    runtime = Qwen3TTSRuntime(
        allow_downloads=False, device="cpu", settings=Qwen3TTSSettings(backend="torch", **settings)
    )
    runtime.snapshot_dir = Mock(return_value="/nonexistent-snapshot")
    runtime._ensure_loaded()
    assert (code_predictor._abstractvoice_predictor, code_predictor._abstractvoice_sampler) == expected
    info = runtime.runtime_info()
    assert info["backend"] == "torch"
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


# ------------------------------------------------------------------ CUDA graphs

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware required")
SETTINGS = [
    dict(do_sample=False, temperature=1.0, top_k=0, top_p=1.0),
    dict(do_sample=True, temperature=0.7, top_k=7, top_p=0.8),
    dict(do_sample=True, temperature=1.0, top_k=0, top_p=1.0),
    dict(do_sample=True, temperature=1.5, top_k=1, top_p=0.2),
]


def last_graph(predictor):
    from abstractvoice.qwen3_tts import predictor as module
    return next(reversed(module._GRAPHS[predictor].values()))


def test_cpu_frames_never_capture_a_graph(predictor):
    from abstractvoice.qwen3_tts import predictor as module
    generate_codebooks(predictor, **arguments(predictor))
    assert module.cuda_graph_state(predictor) == "unused" and not module._GRAPHS.get(predictor)


@needs_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("sampler", ["multinomial", "exponential"])
@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("settings", SETTINGS)
def test_cuda_graph_replays_reference_logits_tokens_and_rng(predictor, dtype, batch, settings, sampler):
    """The capturing call and the replays after it: IDs, every step's logits and the RNG equal the reference."""
    from abstractvoice.qwen3_tts import predictor as module
    predictor.to("cuda", dtype=dtype)
    predictor._abstractvoice_sampler = sampler
    original = predictor.generate
    for seed in [0, 17, 17, 23]:
        torch.manual_seed(seed + 1000)
        kwargs = arguments(predictor, batch)  # fresh inputs: the static input buffer must follow them
        kwargs.update(settings)
        logits = []
        hook = predictor.register_forward_hook(lambda m, a, result: logits.append(result.logits[:, -1].clone()))
        try:
            torch.manual_seed(seed)
            expected = reference(predictor, kwargs)
        finally:
            hook.remove()
        expected_rng = (torch.get_rng_state(), torch.cuda.get_rng_state())
        predictor.generate = Mock(side_effect=AssertionError("unexpected reference fallback"))
        try:
            torch.manual_seed(seed)
            actual = generate_codebooks(predictor, **kwargs)
        finally:
            predictor.generate = original
        assert module.cuda_graph_state(predictor) == "captured"
        assert torch.equal(actual, expected)
        assert torch.equal(torch.get_rng_state(), expected_rng[0])
        assert torch.equal(torch.cuda.get_rng_state(), expected_rng[1])
        assert len(last_graph(predictor).logits) == len(logits) == 3
        for a, b in zip(last_graph(predictor).logits, logits):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert len(module._GRAPHS[predictor]) == 1  # one capture per shape and setting, then replays


@needs_cuda
def test_cuda_graph_capture_failure_falls_back_once_with_rng_intact(predictor, monkeypatch):
    from abstractvoice.qwen3_tts import predictor as module
    predictor.to("cuda")
    kwargs = arguments(predictor)
    torch.manual_seed(5)
    expected = reference(predictor, kwargs)
    expected_rng = torch.cuda.get_rng_state()
    attempts = []
    def broken_graph(*_a, **_kw):
        attempts.append(1)
        raise RuntimeError("capture refused")
    monkeypatch.setattr(module.torch.cuda, "graph", broken_graph)
    warning = Mock()
    monkeypatch.setattr(module.logger, "warning_once", warning)
    for _ in range(2):
        torch.manual_seed(5)  # the warm-up pass ran and drew; its draws must be rolled back
        assert torch.equal(generate_codebooks(predictor, **kwargs), expected)
        assert torch.equal(torch.cuda.get_rng_state(), expected_rng)
    assert len(attempts) == 1 and warning.call_count == 1
    assert "#FALLBACK" in warning.call_args.args[0]
    assert module.cuda_graph_state(predictor).startswith("failed: RuntimeError: capture refused")


@needs_cuda
def test_cuda_graph_is_skipped_for_hooks_and_the_off_switch(predictor):
    from abstractvoice.qwen3_tts import predictor as module
    predictor.to("cuda")
    hook = predictor.model.layers[0].register_forward_hook(lambda *_a: None)
    try:
        generate_codebooks(predictor, **arguments(predictor))
    finally:
        hook.remove()
    predictor._abstractvoice_cuda_graphs = False
    generate_codebooks(predictor, **arguments(predictor))
    assert module.cuda_graph_state(predictor) == "unused"
    predictor._abstractvoice_cuda_graphs = True
    generate_codebooks(predictor, **arguments(predictor))
    assert module.cuda_graph_state(predictor) == "captured"


@needs_cuda
def test_cuda_graph_recaptures_after_the_weights_move(predictor):
    from abstractvoice.qwen3_tts import predictor as module
    predictor.to("cuda")
    kwargs = arguments(predictor)
    kwargs.update(SETTINGS[0])  # greedy: the comparison needs no seeding
    generate_codebooks(predictor, **kwargs)
    first = last_graph(predictor)
    predictor.to(torch.bfloat16)
    kwargs["inputs_embeds"] = kwargs["inputs_embeds"].to(torch.bfloat16)
    assert torch.equal(generate_codebooks(predictor, **kwargs), reference(predictor, kwargs))
    assert last_graph(predictor) is not first and len(module._GRAPHS[predictor]) == 1
    with torch.no_grad():  # same storage, new values: a replay reads them
        predictor.lm_head[0].weight.mul_(-1)
    assert torch.equal(generate_codebooks(predictor, **kwargs), reference(predictor, kwargs))


@needs_cuda
@pytest.mark.parametrize("sampler", ["multinomial", "exponential"])
@pytest.mark.parametrize("step", [0, 2])
def test_cuda_graph_invalid_probabilities_fail_the_frame(predictor, sampler, step):
    predictor.to("cuda")
    predictor._abstractvoice_sampler = sampler
    kwargs = arguments(predictor, batch=2)
    generate_codebooks(predictor, **kwargs)  # captured with valid weights
    predictor.generate = Mock(side_effect=AssertionError("must never retry after sampling"))
    with torch.no_grad():
        predictor.lm_head[step].weight.fill_(float("nan"))
    with pytest.raises(RuntimeError, match="probability tensor"):
        generate_codebooks(predictor, **kwargs)
    predictor.generate.assert_not_called()


@needs_cuda
@pytest.mark.parametrize("exit_order", ["installed", "before-torch-2.15"])
def test_cuda_graph_failing_inside_the_capture_restores_stream_and_rng(predictor, monkeypatch, exit_order):
    """A capture that breaks mid-body (the real torch.cuda.graph) leaves the caller's stream and RNG as they were."""
    from abstractvoice.qwen3_tts import predictor as module
    predictor.to("cuda")
    kwargs = arguments(predictor)
    torch.manual_seed(5)
    expected = reference(predictor, kwargs)
    expected_rng = torch.cuda.get_rng_state()
    real_frame = module._frame
    def frame(*args, **kw):
        result = real_frame(*args, **kw)
        if torch.cuda.is_current_stream_capturing():
            torch.cuda.synchronize()  # a host sync inside the body invalidates the capture
        return result
    monkeypatch.setattr(module, "_frame", frame)
    if exit_order == "before-torch-2.15":
        class ExitBefore215(torch.cuda.graph):
            # torch < 2.15 runs capture_end before unwinding the stream context: when
            # capture_end raises, the graph's private stream stays current.
            def __exit__(self, *args):
                try:
                    return super().__exit__(*args)
                except BaseException:
                    torch.cuda.set_stream(self.capture_stream)
                    raise
        monkeypatch.setattr(module.torch.cuda, "graph", ExitBefore215)
    warning = Mock()
    monkeypatch.setattr(module.logger, "warning_once", warning)
    caller = torch.cuda.current_stream()
    torch.manual_seed(5)
    actual = generate_codebooks(predictor, **kwargs)
    assert torch.cuda.current_stream() == caller
    assert torch.equal(actual, expected)
    assert torch.equal(torch.cuda.get_rng_state(), expected_rng)
    assert module.cuda_graph_state(predictor).startswith("failed: ") and warning.call_count == 1


def load_faked_torch_runtime(monkeypatch, **settings):
    """A Qwen3TTSRuntime "loaded" on the PyTorch backend (CPU) without weights."""
    from types import SimpleNamespace

    from abstractvoice.qwen3_tts import orchestration
    from abstractvoice.qwen3_tts.runtime import Qwen3TTSRuntime, Qwen3TTSSettings

    code_predictor = SimpleNamespace()
    inner = SimpleNamespace(
        talker=SimpleNamespace(code_predictor=code_predictor),
        speech_tokenizer=SimpleNamespace(model=SimpleNamespace(to=lambda **_kw: "codec")),
        device="cpu", tts_model_type="custom_voice", to=lambda _device: None, eval=lambda: None,
    )
    monkeypatch.setattr(
        orchestration.Qwen3TTSModel, "from_pretrained", classmethod(lambda cls, *_a, **_kw: SimpleNamespace(model=inner))
    )
    monkeypatch.setattr("abstractvoice.transformers_floor.require_transformers", lambda *_a, **_kw: None)
    runtime = Qwen3TTSRuntime(
        allow_downloads=False, device="cpu", settings=Qwen3TTSSettings(backend="torch", **settings)
    )
    runtime.snapshot_dir = Mock(return_value="/nonexistent-snapshot")
    runtime._ensure_loaded()
    return runtime


@pytest.mark.parametrize("settings,graphs,expected", [
    ({}, "unused", "multinomial"),
    ({}, "captured", "multinomial (cuda graph: exponential race, per-frame validation)"),
    ({}, "failed: RuntimeError: x", "multinomial"),
    ({"sampler": "exponential"}, "captured", "exponential"),
    ({"predictor": "reference"}, "unused", "multinomial"),
    ({"predictor": "reference"}, "captured", "multinomial"),  # the reference path never replays
])
def test_runtime_info_reports_the_graph_state_and_the_effective_sampler(monkeypatch, settings, graphs, expected):
    from abstractvoice.qwen3_tts import predictor as module
    runtime = load_faked_torch_runtime(monkeypatch, **settings)
    # Unpatched, the real lookup handles the weights-free stand-in model.
    assert runtime.runtime_info()["cuda_graphs"] == "unused"
    monkeypatch.setattr(module, "cuda_graph_state", lambda _predictor: graphs)
    info = runtime.runtime_info()
    assert (info["cuda_graphs"], info["sampler"]) == (graphs, expected)


def test_runtime_info_says_why_its_dtype(monkeypatch):
    from abstractvoice.compute.torch_runtime import TorchRuntimeResolution
    from abstractvoice.qwen3_tts import runtime as module
    from abstractvoice.qwen3_tts.runtime import Qwen3TTSRuntime

    unloaded = Qwen3TTSRuntime(allow_downloads=False, device="cpu", backend="torch").runtime_info()
    assert unloaded["dtype_reason"] is None and unloaded["cuda_graphs"] is None
    assert load_faked_torch_runtime(monkeypatch).runtime_info()["dtype_reason"] == "default on the CPU"
    pinned = Qwen3TTSRuntime(allow_downloads=False, device="cpu", dtype_name="float32", backend="torch")
    pinned._resolve_runtime()
    assert pinned.runtime_info()["dtype_reason"] == "requested"

    def resolve(*, device, dtype_name, allow_cpu_fallback):  # a device whose policy default is float16
        name, reason = ("float16", "default on mps") if dtype_name is None else (dtype_name, "requested")
        return TorchRuntimeResolution(requested_device=device, resolved_device="mps", requested_dtype=dtype_name,
                                      resolved_dtype_name=name, torch_dtype=None, dtype_reason=reason)
    monkeypatch.setattr(module, "resolve_torch_runtime", resolve)
    runtime = Qwen3TTSRuntime(allow_downloads=False, device="mps", backend="torch")
    runtime._resolve_runtime()
    info = runtime.runtime_info()
    assert info["dtype"] == "bfloat16"
    assert info["dtype_reason"] == "float16 overflows Qwen3-TTS, so bfloat16 (the policy chose float16: default on mps)"
