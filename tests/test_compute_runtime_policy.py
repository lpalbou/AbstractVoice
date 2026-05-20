import pytest


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
    monkeypatch.setattr(runtime_module, "best_torch_dtype_name", lambda *, device: "float16")
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
