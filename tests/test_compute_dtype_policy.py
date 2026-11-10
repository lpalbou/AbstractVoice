import sys
import types
from types import SimpleNamespace

import pytest


def _fake_torch(monkeypatch: pytest.MonkeyPatch, capability, *, name="Quadro RTX 5000", bf16_supported=True):
    """Install a `torch` stand-in whose CUDA device has `capability` (an Exception = unreadable).

    Returns the list of device arguments `get_device_capability` was called with.
    """
    seen: list = []

    def get_device_capability(device=None):
        seen.append(device)
        if isinstance(capability, Exception):
            raise capability
        return capability

    torch = types.ModuleType("torch")
    torch.cuda = SimpleNamespace(
        get_device_capability=get_device_capability,
        get_device_name=lambda device=None: name,
        is_bf16_supported=lambda including_emulation=True: bf16_supported,
    )
    torch.float16, torch.bfloat16, torch.float32 = "torch.float16", "torch.bfloat16", "torch.float32"
    monkeypatch.setitem(sys.modules, "torch", torch)
    return seen


@pytest.fixture(autouse=True)
def _no_dtype_override(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ABSTRACTVOICE_TORCH_DTYPE", raising=False)


def test_best_torch_dtype_name_defaults():
    from abstractvoice.compute.dtype import best_torch_dtype_choice, best_torch_dtype_name

    assert best_torch_dtype_name(device="cpu") == "float32"
    assert best_torch_dtype_choice(device="cpu").reason == "default on the CPU"
    # MPS and XPU unchanged.
    assert best_torch_dtype_name(device="mps") == "float16"
    assert best_torch_dtype_name(device="xpu") == "float16"


def test_turing_gets_float32_with_the_reason(monkeypatch: pytest.MonkeyPatch):
    # Quadro RTX 5000, 2026-11-10: emulated bf16 2.2x real time, fp32 1.6x; fp16 crashed.
    from abstractvoice.compute.dtype import best_torch_dtype_choice

    seen = _fake_torch(monkeypatch, (7, 5))
    choice = best_torch_dtype_choice(device="cuda")

    assert choice.name == "float32"
    assert "Quadro RTX 5000 (compute capability 7.5) has no native bfloat16" in choice.reason
    assert seen == [None]  # plain "cuda" = the current device


@pytest.mark.parametrize("capability", [(8, 0), (8, 6), (8, 9), (9, 0), (12, 0)])
def test_ampere_and_newer_get_bfloat16(monkeypatch: pytest.MonkeyPatch, capability):
    from abstractvoice.compute.dtype import best_torch_dtype_choice

    _fake_torch(monkeypatch, capability, name="NVIDIA GeForce RTX 4090")
    choice = best_torch_dtype_choice(device="cuda")

    assert choice.name == "bfloat16"
    assert f"(compute capability {capability[0]}.{capability[1]}) has native bfloat16" in choice.reason


@pytest.mark.parametrize("capability", [(6, 1), (7, 0), (7, 5)])
def test_pre_ampere_never_gets_bfloat16_or_float16(monkeypatch: pytest.MonkeyPatch, capability):
    from abstractvoice.compute.dtype import best_torch_dtype_name

    _fake_torch(monkeypatch, capability)
    assert best_torch_dtype_name(device="cuda") == "float32"


def test_an_explicit_cuda_ordinal_is_the_device_asked(monkeypatch: pytest.MonkeyPatch):
    from abstractvoice.compute.dtype import best_torch_dtype_name

    seen = _fake_torch(monkeypatch, (8, 9))
    assert best_torch_dtype_name(device="cuda:1") == "bfloat16"
    assert best_torch_dtype_name(device="CUDA:0") == "bfloat16"
    assert seen == [1, 0]


def test_an_unreadable_capability_is_float32_with_the_reason(monkeypatch: pytest.MonkeyPatch):
    from abstractvoice.compute.dtype import best_torch_dtype_choice

    _fake_torch(monkeypatch, AssertionError("Torch not compiled with CUDA enabled"))
    choice = best_torch_dtype_choice(device="cuda")

    assert choice.name == "float32"
    assert "compute capability unknown" in choice.reason
    assert "Torch not compiled with CUDA enabled" in choice.reason


def test_best_torch_dtype_name_env_override(monkeypatch: pytest.MonkeyPatch):
    from abstractvoice.compute.dtype import best_torch_dtype_choice, best_torch_dtype_name

    seen = _fake_torch(monkeypatch, (7, 5))
    monkeypatch.setenv("ABSTRACTVOICE_TORCH_DTYPE", "fp16")
    assert best_torch_dtype_name(device="cuda") == "float16"
    monkeypatch.setenv("ABSTRACTVOICE_TORCH_DTYPE", "BF16")
    choice = best_torch_dtype_choice(device="cuda")
    assert choice.name == "bfloat16"
    assert choice.reason == "set by ABSTRACTVOICE_TORCH_DTYPE=BF16"
    assert best_torch_dtype_name(device="cpu") == "bfloat16"
    assert seen == []  # the override wins before any device query


def test_resolve_torch_dtype_accepts_known_names():
    pytest.importorskip("torch")

    from abstractvoice.compute.dtype import resolve_torch_dtype

    dt = resolve_torch_dtype(device="cpu", dtype_name="float32")
    # Avoid importing torch at module import time.
    assert str(dt).endswith("float32")


def test_cuda_without_any_bfloat16_falls_back_to_float32_not_float16(monkeypatch: pytest.MonkeyPatch):
    from abstractvoice.compute.dtype import resolve_torch_dtype

    _fake_torch(monkeypatch, (6, 1), bf16_supported=False)
    assert resolve_torch_dtype(device="cuda", dtype_name="bfloat16") == "torch.float32"
    _fake_torch(monkeypatch, (7, 5), bf16_supported=True)  # emulated: an explicit request is kept
    assert resolve_torch_dtype(device="cuda", dtype_name="bfloat16") == "torch.bfloat16"
