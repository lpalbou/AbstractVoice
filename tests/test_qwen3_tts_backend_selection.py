"""Qwen3-TTS backend policy (ADR 0008) and the MLX settings mapping, torch- and mlx-free.

Nothing here imports torch or mlx or loads a model: the policy is pure Python and the
runtime/adapter/mixin seams are exercised with fakes, so the file runs on CI's Linux
machines exactly as it runs on an Apple silicon laptop.
"""

from __future__ import annotations

import numpy as np
import pytest

from abstractvoice.qwen3_tts import backend as policy
from abstractvoice.qwen3_tts.runtime import (
    BACKEND_ENV,
    MLX_QUANTIZATION_ENV,
    NATIVE_STREAM_MAX_CHARS,
    Qwen3TTSRuntime,
    Qwen3TTSSettings,
    _EdgeFader,
)

@pytest.fixture(autouse=True)
def no_cached_qwen_weights(monkeypatch):
    """The adapter warms a cached checkpoint at construction; these tests use fake runtimes."""
    monkeypatch.setattr("abstractvoice.adapters.tts_qwen3_tts.hf_cached_snapshot_dir", lambda _model_id: None)


DARWIN_ARM = dict(system="darwin", machine="arm64")
LINUX_X86 = dict(system="linux", machine="x86_64")


# --- policy -------------------------------------------------------------------


def test_auto_picks_mlx_only_on_apple_silicon_with_mlx_installed():
    assert policy.select_backend(env={}, importable=True, **DARWIN_ARM) == "mlx"
    assert policy.select_backend(env={}, importable=False, **DARWIN_ARM) == "torch"
    assert policy.select_backend(env={}, importable=True, **LINUX_X86) == "torch"
    assert policy.select_backend(env={}, importable=True, system="darwin", machine="x86_64") == "torch"


def test_environment_override_and_explicit_preference():
    assert policy.select_backend(env={BACKEND_ENV: "torch"}, importable=True, **DARWIN_ARM) == "torch"
    assert policy.select_backend("torch", env={BACKEND_ENV: "mlx"}, importable=True, **DARWIN_ARM) == "torch"
    assert policy.select_backend("auto", env={BACKEND_ENV: "torch"}, importable=True, **DARWIN_ARM) == "torch"
    assert policy.select_backend("mlx", env={}, importable=True, **DARWIN_ARM) == "mlx"


def test_requesting_mlx_where_it_cannot_run_is_a_plain_error():
    with pytest.raises(RuntimeError, match="Apple silicon"):
        policy.select_backend("mlx", env={}, importable=True, **LINUX_X86)
    with pytest.raises(RuntimeError, match="not installed"):
        policy.select_backend(env={BACKEND_ENV: "mlx"}, importable=False, **DARWIN_ARM)
    with pytest.raises(ValueError, match="one of auto, torch, mlx"):
        policy.select_backend("cuda", env={})
    with pytest.raises(ValueError, match="one of auto, torch, mlx"):
        policy.select_backend(env={BACKEND_ENV: "metal"})


def test_describe_backend_selection_never_raises():
    info = policy.describe_backend_selection("mlx", env={})
    assert info["requested"] == "mlx"
    assert info["selected"] in ("mlx", None)
    assert ("error" in info) and ("mlx_available" in info) and ("platform" in info)


# --- settings mapping ----------------------------------------------------------


def test_settings_normalize_backend_and_quantization(monkeypatch):
    monkeypatch.setenv(MLX_QUANTIZATION_ENV, "4bit")  # the setting never reads the environment
    s = Qwen3TTSSettings()
    assert (s.backend, s.mlx_quantization) == ("auto", "auto")
    assert Qwen3TTSSettings(backend=" MLX ", mlx_quantization="bf16").backend == "mlx"
    assert Qwen3TTSSettings(mlx_quantization="4").mlx_quantization == "4bit"
    assert Qwen3TTSSettings(mlx_quantization="none").mlx_quantization == "bf16"
    with pytest.raises(ValueError, match="one of auto, bf16, 8bit, 4bit"):
        Qwen3TTSSettings(mlx_quantization="fp8")
    with pytest.raises(ValueError, match="one of auto, torch, mlx"):
        Qwen3TTSSettings(backend="cuda")
    # The PyTorch-only switches are still accepted next to the MLX settings.
    both = Qwen3TTSSettings(backend="mlx", predictor="reference", sampler="exponential")
    assert (both.predictor, both.sampler) == ("reference", "exponential")
    # Quality presets never touch backend or streaming settings.
    both.apply_quality_preset("low")
    assert (both.backend, both.stream_first_chunk_frames, both.stream_chunk_frames) == ("mlx", 6, 12)
    assert "backend" not in both.generate_kwargs()


def test_runtime_constructor_backend_argument_validates_before_loading():
    with pytest.raises(ValueError, match="one of auto, torch, mlx"):
        Qwen3TTSRuntime(allow_downloads=False, backend="metal")
    runtime = Qwen3TTSRuntime(allow_downloads=False, backend="torch")
    assert runtime.settings.backend == "torch"
    assert runtime.resolve_backend() == "torch"
    assert runtime.supports_native_streaming is False
    info = runtime.runtime_info()
    assert info["backend"] is None and info["backend_requested"] == "torch"
    assert info["backend_selection"]["selected"] == "torch"


def test_runtime_refuses_an_impossible_mlx_request_before_any_weights(monkeypatch):
    monkeypatch.setattr(policy, "mlx_supported_platform", lambda **_kw: False)
    runtime = Qwen3TTSRuntime(allow_downloads=False, backend="mlx")
    runtime.snapshot_dir = lambda *a, **k: pytest.fail("must refuse before resolving a snapshot")
    with pytest.raises(RuntimeError, match="Apple silicon"):
        runtime._ensure_loaded()
    assert runtime.is_loaded is False
    info = runtime.runtime_info()
    assert info["backend"] is None and info["backend_selection"]["error"]


def test_mlx_runtime_info_reports_pytorch_switches_as_not_applied(monkeypatch):
    """The MLX backend accepts predictor/sampler and says what it does with them."""

    class FakeReport:
        quantization = "8bit"
        seconds = {"total": 1.0}
        prime_seconds = 0.5
        quantized_linear_layers = 7

    class FakeMLX:
        model_type = "custom_voice"
        report = FakeReport()

        def supported_speakers(self):
            return ["ryan"]

        def supported_languages(self):
            return ["auto", "english"]

    runtime = Qwen3TTSRuntime(allow_downloads=False, settings=Qwen3TTSSettings(predictor="reference", sampler="exponential"))
    runtime._mlx = FakeMLX()
    runtime._backend = "mlx"
    runtime._resolved_device = "metal"
    runtime._predictor_mode, runtime._sampler = "mlx", "multinomial"
    info = runtime.runtime_info()
    assert info["backend"] == "mlx" and info["device"] == "metal" and info["native_streaming"] is True
    assert info["pytorch_switches"] == {
        "predictor": "reference",
        "sampler": "exponential",
        "applied": False,
        "note": "MLX runs its own fixed-depth codebook loop with categorical (multinomial) sampling",
    }
    assert info["mlx_quantization"] == "8bit" and info["prime_seconds"] == 0.5 and info["quantized_linear_layers"] == 7
    assert runtime.speaker_names() == ["ryan"] and runtime.model_type() == "custom_voice"
    assert runtime.supports_native_streaming is True


# --- utterance-edge fades on a frame stream -------------------------------------


def test_edge_fader_fades_only_the_utterance_edges_and_keeps_samples_contiguous():
    sr = 24000
    fader = _EdgeFader(sample_rate=sr)
    chunks = [np.full(4800, 0.5, dtype=np.float32), np.full(4800, 0.5, dtype=np.float32), np.full(600, 0.5, dtype=np.float32)]
    out = [o for o in (fader.push(c) for c in chunks) if o is not None]
    tail = fader.flush()
    assert tail is not None and len(tail) == fader.fade_samples
    audio = np.concatenate(out + [tail])
    assert len(audio) == sum(len(c) for c in chunks)  # nothing lost, nothing duplicated
    assert audio[0] == 0.0 and audio[-1] == 0.0  # de-clicked edges
    n = fader.fade_samples
    assert np.all(audio[n:-n] == 0.5)  # the interior, including every chunk seam, is untouched
    assert fader.flush() is None


# --- adapter and mixin seams -----------------------------------------------------


class _NativeRuntime:
    """A runtime that streams frames of ONE generation per call (what the MLX backend does)."""

    model_id = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
    is_loaded = True
    backend = "mlx"
    supports_native_streaming = True

    def __init__(self):
        self.settings = Qwen3TTSSettings()
        self.calls = []

    def model_type(self):
        return "custom_voice"

    def speaker_names(self):
        return ["ryan", "serena"]

    def resolve_backend(self):
        return "mlx"

    def runtime_info(self):
        return {"backend": "mlx", "device": "metal"}

    def stream_custom_voice(self, text, **kwargs):
        self.calls.append((text, kwargs))
        for _ in range(3):
            yield np.full(1920, 0.2, dtype=np.float32), 24000

    def synthesize_custom_voice(self, text, **kwargs):
        chunks = [c for c, _ in self.stream_custom_voice(text, **kwargs)]
        return np.concatenate(chunks), 24000


def test_adapter_streams_one_generation_per_reply_on_a_native_runtime():
    from abstractvoice.adapters.tts_qwen3_tts import Qwen3TTSAdapter

    runtime = _NativeRuntime()
    adapter = Qwen3TTSAdapter(runtime=runtime)
    assert adapter.native_frame_streaming is True
    assert adapter.get_max_chars() == NATIVE_STREAM_MAX_CHARS
    assert adapter.execution_device() == {"device": "metal", "reason": "Apple GPU (MLX)"}
    text = "First sentence. Second sentence keeps the voice. Third one too! " * 6
    assert len(text) < NATIVE_STREAM_MAX_CHARS
    chunks = list(adapter.synthesize_to_audio_chunks_with_instructions(text, instructions="Calm"))
    assert len(chunks) == 3 and all(sr == 24000 for _, sr in chunks)
    assert len(runtime.calls) == 1  # the whole multi-sentence reply is ONE generation
    assert runtime.calls[0][0] == text.strip() and runtime.calls[0][1]["instruct"] == "Calm"
    assert runtime.calls[0][1]["speaker"] == "ryan"
    # Beyond the native bound the text is cut at sentence boundaries into independent generations.
    long_text = "A sentence that is long enough to matter. " * 40
    list(adapter.synthesize_to_audio_chunks(long_text))
    assert len(runtime.calls) > 1
    assert all(len(t) <= NATIVE_STREAM_MAX_CHARS for t, _ in runtime.calls[1:])


def test_mixin_keeps_native_frame_chunks_whole_and_reports_the_unit():
    from abstractvoice.adapters.tts_qwen3_tts import Qwen3TTSAdapter
    from abstractvoice.vm.tts_mixin import TtsMixin

    class VM(TtsMixin):
        def __init__(self, adapter):
            self.tts_adapter = adapter
            self.language = "en"
            self.speed = 1.0
            self.metrics = None

        def _set_last_tts_metrics(self, metrics):
            self.metrics = metrics

    runtime = _NativeRuntime()
    adapter = Qwen3TTSAdapter(runtime=runtime)
    vm = VM(adapter)
    reply = "One sentence here. Another sentence there. And a third, to be sure."
    chunks = list(vm.speak_to_audio_chunks(reply))
    # Quiet-but-not-silent frames are NOT trimmed: they are contiguous pieces of one waveform.
    assert len(chunks) == 3 and all(len(c) == 1920 for c, _ in chunks)
    assert len(runtime.calls) == 1 and runtime.calls[0][0] == reply
    m = vm.metrics
    assert m["streaming_unit"] == "frames" and m["segments"] == 1 and m["segment_chars"] == [len(reply)]
    assert m["chunks"] == 3 and m["device"] == "metal" and m["device_reason"] == "Apple GPU (MLX)"
    assert m["ttfb_s"] is not None


def test_mixin_keeps_segment_streaming_for_non_native_adapters():
    from abstractvoice.vm.tts_mixin import TtsMixin

    class Adapter:
        engine_id = "fake"

        def __init__(self):
            self.calls = []

        def is_available(self):
            return True

        def get_max_chars(self):
            return 300

        def synthesize_to_audio_chunks(self, text):
            self.calls.append(text)
            yield np.full(240, 0.1, dtype=np.float32), 24000

    class VM(TtsMixin):
        def __init__(self, adapter):
            self.tts_adapter = adapter
            self.language = "en"
            self.speed = 1.0
            self.metrics = None

        def _set_last_tts_metrics(self, metrics):
            self.metrics = metrics

    adapter = Adapter()
    vm = VM(adapter)
    list(vm.speak_to_audio_chunks("The first sentence is spoken alone. Then the rest follows in a group of its own."))
    assert len(adapter.calls) == 2 and vm.metrics["streaming_unit"] == "segments"


def test_cloning_engine_streams_the_whole_text_on_a_native_runtime(tmp_path):
    from abstractvoice.cloning.engine_qwen3_tts import Qwen3TTSVoiceCloningEngine

    class Runtime:
        supports_native_streaming = True
        calls = []

        def build_clone_prompt(self, **kwargs):
            return "prompt"

        def stream_clone(self, text, *, clone_prompt, language):
            Runtime.calls.append((text, clone_prompt, language))
            yield np.full(1920, 0.1, dtype=np.float32), 24000
            yield np.full(1920, 0.1, dtype=np.float32), 24000

    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"RIFF")
    engine = Qwen3TTSVoiceCloningEngine(allow_downloads=False)
    engine._runtime = Runtime()
    text = "A cloned voice speaks this sentence. And this one, without drifting. " * 3
    chunks = list(engine.infer_to_audio_chunks(text=text, reference_paths=[str(ref)], reference_text="Reference", max_chars=120))
    assert len(chunks) == 2 and Runtime.calls == [(text.strip(), "prompt", None)]


def test_engine_runtime_status_follows_the_backend_policy(monkeypatch):
    from abstractvoice import engine_runtime

    monkeypatch.setattr(policy, "select_backend", lambda *a, **k: "mlx")
    status = engine_runtime.engine_runtime_status("qwen3-tts")
    assert status.required_modules == ("mlx", "tokenizers", "soundfile")
    monkeypatch.setattr(policy, "select_backend", lambda *a, **k: "torch")
    status = engine_runtime.engine_runtime_status("qwen3-tts")
    assert status.required_modules == ("torch", "transformers", "soundfile")



# --- review fixes ---------------------------------------------------------------


def test_mlx_quantization_environment_is_read_only_when_mlx_loads():
    assert policy.resolve_mlx_quantization("auto", env={}) == "8bit"
    assert policy.resolve_mlx_quantization("auto", env={MLX_QUANTIZATION_ENV: "4bit"}) == "4bit"
    assert policy.resolve_mlx_quantization("bf16", env={MLX_QUANTIZATION_ENV: "4bit"}) == "bf16"  # explicit wins
    with pytest.raises(ValueError, match="ABSTRACTVOICE_QWEN3_TTS_MLX_QUANTIZATION='fp8'"):
        policy.resolve_mlx_quantization("auto", env={MLX_QUANTIZATION_ENV: "fp8"})


def test_a_stray_mlx_quantization_variable_never_breaks_the_torch_backend(monkeypatch):
    monkeypatch.setenv(MLX_QUANTIZATION_ENV, "fp8")
    runtime = Qwen3TTSRuntime(allow_downloads=False, backend="torch")
    assert runtime.settings.mlx_quantization == "auto"


def test_runtime_backend_argument_does_not_mutate_the_callers_settings():
    shared = Qwen3TTSSettings()
    Qwen3TTSRuntime(allow_downloads=False, settings=shared, backend="torch")
    assert shared.backend == "auto"


def test_mlx_version_floor_is_a_plain_error():
    assert policy.require_mlx_version(installed=policy.MLX_MIN_VERSION) == policy.MLX_MIN_VERSION
    assert policy.require_mlx_version(installed="0.32.4") == "0.32.4"
    with pytest.raises(RuntimeError, match=r"mlx 0\.31\.2 or newer.*found 0\.26\.1"):
        policy.require_mlx_version(installed="0.26.1")


def test_a_suspended_stream_makes_other_calls_fail_clearly_instead_of_hanging():
    import threading

    runtime = Qwen3TTSRuntime(allow_downloads=False, backend="torch")
    runtime.LOCK_TIMEOUT_S = 0.2
    runtime.UNLOAD_LOCK_TIMEOUT_S = 0.2
    holder_started, release = threading.Event(), threading.Event()

    def hold():
        with runtime._owned():
            holder_started.set()
            release.wait(5)

    thread = threading.Thread(target=hold)
    thread.start()
    holder_started.wait(5)
    try:
        with pytest.raises(RuntimeError, match=r"close\(\)"):
            with runtime._owned():
                pass
        with pytest.raises(RuntimeError, match="unload timed out"):
            runtime.unload()
    finally:
        release.set()
        thread.join(5)
    with runtime._owned():  # free again once the holder is done
        pass


def test_explicit_mlx_request_that_cannot_run_is_reported_on_every_surface(monkeypatch):
    from abstractvoice import engine_runtime
    from abstractvoice.adapters.tts_qwen3_tts import Qwen3TTSAdapter

    monkeypatch.setenv(BACKEND_ENV, "mlx")
    monkeypatch.setattr(policy, "mlx_supported_platform", lambda **_kw: False)

    adapter = Qwen3TTSAdapter(allow_downloads=False, auto_load=False)
    assert adapter.is_available() is False
    assert "Apple silicon" in (adapter.get_unavailable_reason() or "")

    status = engine_runtime.engine_runtime_status("qwen3-tts")
    assert status.installed is False
    assert "Apple silicon" in (status.reason or "")


def test_a_loaded_runtime_is_available_whatever_the_platform_policy_says(monkeypatch):
    """CI runs on Linux without torch: a runtime that is already loaded (here a fake MLX one)
    must not be reported unavailable because the PyTorch packages are absent."""
    import importlib.util

    from abstractvoice.adapters.tts_qwen3_tts import Qwen3TTSAdapter

    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name, *a: None if name in ("torch", "transformers", "mlx") else real_find_spec(name, *a)
    )
    monkeypatch.setattr(policy, "mlx_supported_platform", lambda **_kw: False)
    assert Qwen3TTSAdapter(runtime=_NativeRuntime()).is_available() is True
