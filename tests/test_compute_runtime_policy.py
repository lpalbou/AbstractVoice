import os
import sys

import pytest

from abstractvoice.compute.dtype import TorchDtypeChoice


def test_resolve_torch_runtime_auto_uses_best_device(monkeypatch: pytest.MonkeyPatch) -> None:
    from abstractvoice.compute import torch_runtime as runtime_module

    monkeypatch.setattr(runtime_module, "best_torch_device", lambda: "mps")
    monkeypatch.setattr(runtime_module, "_device_available", lambda device: (True, None))
    monkeypatch.setattr(
        runtime_module,
        "resolve_torch_dtype",
        lambda *, device, dtype_name=None: "torch.float16" if device == "mps" else "torch.float32",
    )

    runtime = runtime_module.resolve_torch_runtime(device="auto")

    assert runtime.requested_device == "auto"
    assert runtime.resolved_device == "mps"
    assert runtime.resolved_dtype_name == "float16"
    assert runtime.used_fallback is False
    assert runtime.fallback_reason is None


def test_resolve_torch_runtime_raises_when_explicit_device_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from abstractvoice.compute import torch_runtime as runtime_module

    monkeypatch.setattr(runtime_module, "_device_available", lambda device: (False, "MPS unavailable"))

    with pytest.raises(RuntimeError, match="MPS unavailable"):
        runtime_module.resolve_torch_runtime(device="mps", allow_cpu_fallback=False)


def test_resolve_torch_runtime_can_fallback_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    from abstractvoice.compute import torch_runtime as runtime_module

    def fake_device_available(device: str):
        if device == "mps":
            return False, "MPS unavailable"
        return True, None

    monkeypatch.setattr(runtime_module, "_device_available", fake_device_available)
    monkeypatch.setattr(
        runtime_module,
        "resolve_torch_dtype",
        lambda *, device, dtype_name=None: "torch.float32" if device == "cpu" else "torch.float16",
    )

    runtime = runtime_module.resolve_torch_runtime(
        device="mps",
        dtype_name="float16",
        allow_cpu_fallback=True,
    )

    assert runtime.requested_device == "mps"
    assert runtime.resolved_device == "cpu"
    assert runtime.resolved_dtype_name == "float32"
    assert runtime.used_fallback is True
    assert "Falling back to CPU" in str(runtime.fallback_reason)


def test_resolve_torch_runtime_preserves_explicit_cuda_ordinal(monkeypatch: pytest.MonkeyPatch) -> None:
    from abstractvoice.compute import torch_runtime as runtime_module

    seen: dict[str, object] = {}

    monkeypatch.setattr(runtime_module, "_device_available", lambda device: (True, None))

    def fake_resolve_torch_dtype(*, device, dtype_name=None):
        seen["device"] = device
        seen["dtype_name"] = dtype_name
        return "torch.float16"

    monkeypatch.setattr(runtime_module, "resolve_torch_dtype", fake_resolve_torch_dtype)

    runtime = runtime_module.resolve_torch_runtime(device="cuda:1", dtype_name="float16")

    assert runtime.requested_device == "cuda:1"
    assert runtime.resolved_device == "cuda:1"
    assert seen == {"device": "cuda:1", "dtype_name": "float16"}


def test_resolve_torch_runtime_keeps_legacy_auto_dtype_behavior(monkeypatch: pytest.MonkeyPatch) -> None:
    from abstractvoice.compute import torch_runtime as runtime_module

    monkeypatch.setattr(runtime_module, "_device_available", lambda device: (True, None))
    monkeypatch.setattr(runtime_module, "best_torch_dtype_choice", lambda *, device: TorchDtypeChoice("float16", "test"))
    monkeypatch.setattr(runtime_module, "resolve_torch_dtype", lambda *, device, dtype_name=None: "torch.float16")

    runtime = runtime_module.resolve_torch_runtime(device="mps", dtype_name="auto")

    assert runtime.requested_dtype == "auto"
    assert runtime.resolved_dtype_name == "float16"
    assert runtime.torch_dtype == "torch.float16"


def test_resolve_torch_runtime_invalid_dtype_degrades_to_engine_default(monkeypatch: pytest.MonkeyPatch) -> None:
    from abstractvoice.compute import torch_runtime as runtime_module

    monkeypatch.setattr(runtime_module, "_device_available", lambda device: (True, None))

    def fail_resolve_torch_dtype(*, device, dtype_name=None):
        raise ValueError(f"Unsupported dtype_name: {dtype_name}")

    monkeypatch.setattr(runtime_module, "resolve_torch_dtype", fail_resolve_torch_dtype)

    runtime = runtime_module.resolve_torch_runtime(device="cpu", dtype_name="weird")

    assert runtime.requested_dtype == "weird"
    assert runtime.resolved_dtype_name == "weird"
    assert runtime.torch_dtype is None


@pytest.mark.skipif(sys.version_info < (3, 10), reason="F5/OpenF5 is unsupported on Python < 3.10")
def test_openf5_runtime_resolution_uses_shared_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from abstractvoice.cloning import engine_f5 as f5_module

    seen: dict[str, object] = {}

    def fake_resolve_torch_runtime(*, device="auto", dtype_name=None, allow_cpu_fallback=False):
        seen.update(
            {
                "device": device,
                "dtype_name": dtype_name,
                "allow_cpu_fallback": allow_cpu_fallback,
            }
        )
        return SimpleNamespace(
            requested_device=str(device),
            resolved_device="mps",
            requested_dtype=dtype_name,
            resolved_dtype_name="float16",
            torch_dtype="torch.float16",
            used_fallback=False,
            fallback_reason=None,
        )

    monkeypatch.setattr(f5_module, "resolve_torch_runtime", fake_resolve_torch_runtime)

    engine = f5_module.F5TTSVoiceCloningEngine(device="auto")
    runtime = engine._resolve_runtime()

    assert runtime.resolved_device == "mps"
    assert seen == {
        "device": "auto",
        "dtype_name": None,
        "allow_cpu_fallback": True,
    }


@pytest.mark.skipif(sys.version_info < (3, 10), reason="F5/OpenF5 is unsupported on Python < 3.10")
def test_openf5_runtime_info_surfaces_fallback_metadata() -> None:
    from abstractvoice.cloning.engine_f5 import F5TTSVoiceCloningEngine

    engine = F5TTSVoiceCloningEngine(device="auto")
    engine._f5_device = "cpu"
    engine._used_fallback = True
    engine._fallback_reason = "Falling back to CPU because MPS load failed."

    info = engine.runtime_info()

    assert info["requested_device"] == "auto"
    assert info["resolved_device"] == "cpu"
    assert info["used_fallback"] is True
    assert info["fallback_reason"] == "Falling back to CPU because MPS load failed."


def test_transformers_asr_runtime_resolution_uses_shared_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from abstractvoice.adapters import stt_transformers_asr as asr_module

    seen: dict[str, object] = {}

    def fake_ensure_loaded(self) -> None:
        _ = self

    def fake_resolve_torch_runtime(*, device="auto", dtype_name=None, allow_cpu_fallback=False):
        seen.update(
            {
                "device": device,
                "dtype_name": dtype_name,
                "allow_cpu_fallback": allow_cpu_fallback,
            }
        )
        return SimpleNamespace(
            requested_device=str(device),
            resolved_device="mps",
            requested_dtype=dtype_name,
            resolved_dtype_name="float16",
            torch_dtype="torch.float16",
            used_fallback=False,
            fallback_reason=None,
        )

    monkeypatch.setattr(asr_module.TransformersASRAdapter, "_ensure_loaded", fake_ensure_loaded)
    monkeypatch.setattr(asr_module, "resolve_torch_runtime", fake_resolve_torch_runtime)

    adapter = asr_module.TransformersASRAdapter(
        model_id="openai/whisper-large-v3",
        device="auto",
        dtype="float16",
    )
    runtime = adapter._resolve_runtime()

    assert runtime.resolved_device == "mps"
    assert seen == {
        "device": "auto",
        "dtype_name": "float16",
        "allow_cpu_fallback": True,
    }


def test_transformers_asr_records_unavailable_reason_from_init_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    from abstractvoice.adapters import stt_transformers_asr as asr_module

    def fail_ensure_loaded(self) -> None:
        _ = self
        raise RuntimeError("MPS load failed")

    monkeypatch.setattr(asr_module.TransformersASRAdapter, "_ensure_loaded", fail_ensure_loaded)

    adapter = asr_module.TransformersASRAdapter(
        model_id="openai/whisper-large-v3",
        device="auto",
    )

    assert adapter.is_available() is False
    assert adapter.get_unavailable_reason() == "MPS load failed"
    assert adapter.get_info()["runtime"]["load_error"] == "MPS load failed"


def test_resolve_torch_runtime_on_mps_makes_transformers_load_weights_sequentially(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import abstractvoice.compute.torch_runtime as runtime_module

    monkeypatch.delenv("HF_DEACTIVATE_ASYNC_LOAD", raising=False)
    monkeypatch.setattr(runtime_module, "_device_available", lambda device: (True, None))
    monkeypatch.setattr(runtime_module, "best_torch_dtype_choice", lambda *, device: TorchDtypeChoice("float16", "test"))
    monkeypatch.setattr(runtime_module, "resolve_torch_dtype", lambda *, device, dtype_name=None: "torch.float16")

    runtime_module.resolve_torch_runtime(device="mps")

    assert os.environ.get("HF_DEACTIVATE_ASYNC_LOAD") == "1"


@pytest.mark.parametrize("device", ["cpu", "cuda", "cuda:1"])
def test_resolve_torch_runtime_leaves_transformers_loading_alone_off_mps(
    monkeypatch: pytest.MonkeyPatch, device: str
) -> None:
    import abstractvoice.compute.torch_runtime as runtime_module

    monkeypatch.delenv("HF_DEACTIVATE_ASYNC_LOAD", raising=False)
    monkeypatch.setattr(runtime_module, "_device_available", lambda device: (True, None))
    monkeypatch.setattr(runtime_module, "best_torch_dtype_choice", lambda *, device: TorchDtypeChoice("float32", "test"))
    monkeypatch.setattr(runtime_module, "resolve_torch_dtype", lambda *, device, dtype_name=None: "torch.float32")

    runtime_module.resolve_torch_runtime(device=device)

    assert "HF_DEACTIVATE_ASYNC_LOAD" not in os.environ


def test_resolve_torch_runtime_respects_an_explicit_async_load_choice(monkeypatch: pytest.MonkeyPatch) -> None:
    import abstractvoice.compute.torch_runtime as runtime_module

    monkeypatch.setenv("HF_DEACTIVATE_ASYNC_LOAD", "0")
    monkeypatch.setattr(runtime_module, "_device_available", lambda device: (True, None))
    monkeypatch.setattr(runtime_module, "best_torch_dtype_choice", lambda *, device: TorchDtypeChoice("float16", "test"))
    monkeypatch.setattr(runtime_module, "resolve_torch_dtype", lambda *, device, dtype_name=None: "torch.float16")

    assert runtime_module.guard_metal_weight_loading("mps") is False
    runtime_module.resolve_torch_runtime(device="mps")

    assert os.environ["HF_DEACTIVATE_ASYNC_LOAD"] == "0"


def _fake_cuda_torch(monkeypatch: pytest.MonkeyPatch, capability, name="Quadro RTX 5000"):
    """A `torch` stand-in with one CUDA device of `capability`; the device counts as available."""
    import types
    from types import SimpleNamespace

    import abstractvoice.compute.torch_runtime as runtime_module

    torch = types.ModuleType("torch")
    torch.cuda = SimpleNamespace(
        get_device_capability=lambda device=None: capability,
        get_device_name=lambda device=None: name,
        is_bf16_supported=lambda including_emulation=True: True,
    )
    torch.float16, torch.bfloat16, torch.float32 = "torch.float16", "torch.bfloat16", "torch.float32"
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(runtime_module, "_device_available", lambda device: (True, None))
    monkeypatch.delenv("ABSTRACTVOICE_TORCH_DTYPE", raising=False)
    # Recorded so a test on "mps" cannot leak the variable its guard sets.
    monkeypatch.setenv("HF_DEACTIVATE_ASYNC_LOAD", "0")


def test_turing_runtime_is_float32_and_says_why(monkeypatch: pytest.MonkeyPatch) -> None:
    from abstractvoice.compute import resolve_torch_runtime

    _fake_cuda_torch(monkeypatch, (7, 5))
    runtime = resolve_torch_runtime(device="cuda")
    info = runtime.to_runtime_info()

    assert runtime.resolved_dtype_name == "float32"
    assert info["resolved_dtype"] == "float32"
    assert "compute capability 7.5) has no native bfloat16" in info["dtype_reason"]


@pytest.mark.parametrize("capability", [(8, 0), (8, 9)])
def test_ampere_and_newer_runtime_is_bfloat16(monkeypatch: pytest.MonkeyPatch, capability) -> None:
    from abstractvoice.compute import resolve_torch_runtime

    _fake_cuda_torch(monkeypatch, capability, name="NVIDIA A100")
    runtime = resolve_torch_runtime(device="cuda:0")

    assert runtime.resolved_device == "cuda:0"
    assert runtime.resolved_dtype_name == "bfloat16"
    assert "has native bfloat16" in runtime.to_runtime_info()["dtype_reason"]


def test_runtime_dtype_reason_for_requests_overrides_and_the_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    from abstractvoice.compute import resolve_torch_runtime

    _fake_cuda_torch(monkeypatch, (7, 5))
    explicit = resolve_torch_runtime(device="cuda", dtype_name="bfloat16")
    assert (explicit.resolved_dtype_name, explicit.dtype_reason) == ("bfloat16", "requested")

    cpu = resolve_torch_runtime(device="cpu")
    assert (cpu.resolved_dtype_name, cpu.dtype_reason) == ("float32", "default on the CPU")

    monkeypatch.setenv("ABSTRACTVOICE_TORCH_DTYPE", "bf16")
    forced = resolve_torch_runtime(device="cuda")
    assert (forced.resolved_dtype_name, forced.dtype_reason) == ("bfloat16", "set by ABSTRACTVOICE_TORCH_DTYPE=bf16")


@pytest.mark.parametrize(
    "device, capability, dtype_name, expected",
    [
        ("cuda", (7, 5), None, "float32"),  # Turing: the shared policy's float32
        ("cuda", (8, 9), None, "bfloat16"),  # Ada: native bf16
        ("cpu", (7, 5), None, "float32"),
        ("mps", (7, 5), None, "bfloat16"),  # unchanged: the shared fp16 default overflows Qwen3-TTS
        ("cuda", (7, 5), "float16", "float16"),  # an explicit pin is the caller's choice
        ("cuda", (8, 9), "auto", "bfloat16"),
    ],
)
def test_qwen3_tts_takes_the_shared_dtype_policy(
    monkeypatch: pytest.MonkeyPatch, device, capability, dtype_name, expected
) -> None:
    from abstractvoice.qwen3_tts.runtime import Qwen3TTSRuntime

    _fake_cuda_torch(monkeypatch, capability)
    runtime = Qwen3TTSRuntime(device=device, dtype_name=dtype_name, backend="torch")._resolve_runtime()

    assert runtime.resolved_device == device
    assert runtime.resolved_dtype_name == expected


def test_qwen3_tts_never_defaults_to_float16_even_when_the_environment_asks(monkeypatch: pytest.MonkeyPatch) -> None:
    from abstractvoice.qwen3_tts.runtime import Qwen3TTSRuntime

    _fake_cuda_torch(monkeypatch, (8, 9))
    monkeypatch.setenv("ABSTRACTVOICE_TORCH_DTYPE", "fp16")
    assert Qwen3TTSRuntime(device="cuda", backend="torch")._resolve_runtime().resolved_dtype_name == "bfloat16"
