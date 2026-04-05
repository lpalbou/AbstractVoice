import pytest


def test_best_torch_dtype_name_defaults():
    from abstractvoice.compute.dtype import best_torch_dtype_name

    assert best_torch_dtype_name(device="cpu") == "float32"
    assert best_torch_dtype_name(device="mps") == "float16"
    assert best_torch_dtype_name(device="cuda") == "bfloat16"


def test_best_torch_dtype_name_env_override(monkeypatch: pytest.MonkeyPatch):
    from abstractvoice.compute.dtype import best_torch_dtype_name

    monkeypatch.setenv("ABSTRACTVOICE_TORCH_DTYPE", "fp16")
    assert best_torch_dtype_name(device="cuda") == "float16"


def test_resolve_torch_dtype_accepts_known_names():
    from abstractvoice.compute.dtype import resolve_torch_dtype

    dt = resolve_torch_dtype(device="cpu", dtype_name="float32")
    # Avoid importing torch at module import time.
    assert str(dt).endswith("float32")

