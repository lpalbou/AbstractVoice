"""Torch dtype selection helpers.

We keep dtype selection centralized so new torch-based engines (TTS/cloning)
behave consistently across platforms.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# NVIDIA GPUs compute natively in bfloat16 from Ampere (compute capability 8.0)
# on. Older cards (Turing 7.5, Volta 7.0, Pascal 6.x) run bfloat16 emulated:
# measured on a Quadro RTX 5000 (7.5), Qwen3-TTS 0.6B took 2.2x real time in
# bfloat16 (emulated `magma_sgemmEx` matmuls, math-only attention) and 1.6x in
# float32 (efficient attention kernel). float16 is not the answer there either:
# it crashed that generation (a device-side assert in the rotary embedding).
NATIVE_BF16_MIN_CUDA_CAPABILITY = (8, 0)


@dataclass(frozen=True)
class TorchDtypeChoice:
    """A recommended dtype name and the short reason it was chosen."""

    name: str
    reason: str
    # True when ABSTRACTVOICE_TORCH_DTYPE chose it: an explicit knob, not the device default.
    explicit: bool = False


def _device_base(device: str) -> str:
    text = str(device or "").strip().lower()
    if text.startswith("cuda"):
        return "cuda"
    if text.startswith("mps"):
        return "mps"
    if text.startswith("xpu"):
        return "xpu"
    return text


def _cuda_ordinal(device: str) -> int | None:
    """The explicit ordinal of `cuda:N`, or None for plain `cuda` (the current device)."""
    text = str(device or "").strip().lower()
    if ":" not in text:
        return None
    try:
        return int(text.split(":", 1)[1])
    except ValueError:
        return None


def _cuda_bf16_choice(device: str) -> TorchDtypeChoice:
    index = _cuda_ordinal(device)
    try:
        import torch

        major, minor = (int(part) for part in torch.cuda.get_device_capability(index))
    except Exception as e:  # noqa: BLE001 - no torch, no CUDA runtime, bad ordinal
        return TorchDtypeChoice(
            "float32",
            f"CUDA compute capability unknown ({type(e).__name__}: {e}); "
            "bfloat16 is chosen only on GPUs with native bfloat16",
        )
    try:
        label = str(torch.cuda.get_device_name(index))
    except Exception:  # noqa: BLE001 - the name is decoration only
        label = "CUDA device"
    capability = f"{label} (compute capability {major}.{minor})"
    if (major, minor) >= NATIVE_BF16_MIN_CUDA_CAPABILITY:
        return TorchDtypeChoice("bfloat16", f"{capability} has native bfloat16")
    return TorchDtypeChoice(
        "float32",
        f"{capability} has no native bfloat16 (8.0, Ampere, and newer have it): "
        "float32 runs faster than emulated bfloat16, and float16 is unstable",
    )


def best_torch_dtype_choice(*, device: str) -> TorchDtypeChoice:
    """Return the recommended torch dtype for `device` with the reason.

    Honors env var `ABSTRACTVOICE_TORCH_DTYPE` when set. On CUDA, bfloat16 only
    when the device (`cuda:N` or the current one) computes it natively, else
    float32; never float16.
    """
    raw = (os.environ.get("ABSTRACTVOICE_TORCH_DTYPE") or "").strip()
    forced = raw.lower()
    if forced:
        reason = f"set by ABSTRACTVOICE_TORCH_DTYPE={raw}"
        # normalize common spellings
        if forced in ("fp16", "float16", "half"):
            return TorchDtypeChoice("float16", reason, explicit=True)
        if forced in ("bf16", "bfloat16"):
            return TorchDtypeChoice("bfloat16", reason, explicit=True)
        if forced in ("fp32", "float32"):
            return TorchDtypeChoice("float32", reason, explicit=True)
        return TorchDtypeChoice(forced, reason, explicit=True)

    dev = _device_base(str(device or ""))
    if dev == "cuda":
        return _cuda_bf16_choice(str(device))
    if dev == "mps":
        return TorchDtypeChoice("float16", "default on the Apple GPU (MPS)")
    if dev == "xpu":
        # Conservative; engine can override.
        return TorchDtypeChoice("float16", "default on Intel XPU")
    return TorchDtypeChoice("float32", "default on the CPU")


def best_torch_dtype_name(*, device: str) -> str:
    """Return a recommended torch dtype name for `device` (see `best_torch_dtype_choice`)."""
    return best_torch_dtype_choice(device=device).name


def resolve_torch_dtype(*, device: str, dtype_name: str | None = None):
    """Resolve a torch dtype object for a device.

    This imports torch lazily so importing AbstractVoice stays lightweight.
    """
    try:
        import torch
    except Exception as e:  # pragma: no cover
        raise RuntimeError("torch is required to resolve a torch dtype") from e

    raw_name = (dtype_name or "").strip().lower()
    if not raw_name or raw_name == "auto":
        choice = best_torch_dtype_choice(device=device)
        name, request = choice.name, (choice.reason if choice.explicit else None)
    else:
        name, request = raw_name, f"dtype_name={dtype_name!r}"
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype_name: {dtype_name}")

    dt = mapping[name]

    # A CUDA device that cannot run bfloat16 at all (not even emulated) gets
    # float32: float16 is unstable for these models (see above). An explicit
    # request rewritten this way says so (ADR 0005: explicit knobs come first).
    if _device_base(str(device)) == "cuda" and dt == torch.bfloat16:
        try:
            if hasattr(torch.cuda, "is_bf16_supported") and not torch.cuda.is_bf16_supported():
                if request:
                    logger.warning("bfloat16 requested (%s) but %s cannot run bfloat16; using float32", request, device)
                return torch.float32
        except Exception:
            # Best-effort; keep requested dtype.
            return dt

    return dt
