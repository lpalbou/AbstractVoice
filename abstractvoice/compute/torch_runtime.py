"""Shared torch runtime resolution helpers.

These helpers sit above raw device/dtype defaults so engines can expose
explicit fallback behavior instead of quietly moving to CPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .device import best_torch_device
from .dtype import best_torch_dtype_name, resolve_torch_dtype


def _norm_device_name(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text or text == "auto":
        return "auto"
    return text


def _device_base(device: Any) -> str:
    text = _norm_device_name(device)
    if text.startswith("cuda"):
        return "cuda"
    if text.startswith("mps"):
        return "mps"
    if text.startswith("xpu"):
        return "xpu"
    return text


def _device_index(device: str, prefix: str) -> int | None:
    text = _norm_device_name(device)
    head = f"{prefix}:"
    if not text.startswith(head):
        return None
    try:
        return int(text.split(":", 1)[1])
    except Exception:
        return None


def _device_available(device: str) -> tuple[bool, str | None]:
    raw = _norm_device_name(device)
    dev = _device_base(raw)
    if dev == "cpu":
        return True, None

    try:
        import torch
    except Exception as e:  # pragma: no cover
        return False, f"torch import failed: {e}"

    if dev == "cuda":
        try:
            if torch.cuda.is_available():
                idx = _device_index(raw, "cuda")
                if idx is not None:
                    try:
                        count = int(torch.cuda.device_count())
                    except Exception:
                        count = None
                    if count is not None and (idx < 0 or idx >= count):
                        return False, f"Requested CUDA device '{raw}' is unavailable ({count} visible device(s))."
                return True, None
        except Exception:
            pass
        return False, "CUDA is not available in the active torch runtime."

    if dev == "mps":
        try:
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return True, None
        except Exception:
            pass
        return False, "MPS is not available in the active torch runtime."

    if dev == "xpu":
        try:
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                return True, None
        except Exception:
            pass
        return False, "XPU is not available in the active torch runtime."

    return False, f"Unsupported torch device '{dev}'."


def looks_like_torch_device_error(exc: Exception, *, attempted_device: str) -> bool:
    text = f"{type(exc).__name__}: {exc}".strip().lower()
    base = _device_base(attempted_device)
    tokens = {
        base,
        "cuda",
        "mps",
        "xpu",
        "device type",
        "same device",
        "unsupported autocast device_type",
        "not implemented for",
        "backend",
        "out of memory",
        "insufficient memory",
    }
    return any(token and token in text for token in tokens)


@dataclass(frozen=True)
class TorchRuntimeResolution:
    requested_device: str
    resolved_device: str
    requested_dtype: str | None
    resolved_dtype_name: str
    torch_dtype: Any
    used_fallback: bool = False
    fallback_reason: str | None = None

    def to_runtime_info(self) -> dict[str, Any]:
        return {
            "requested_device": self.requested_device,
            "resolved_device": self.resolved_device,
            "requested_dtype": self.requested_dtype,
            "resolved_dtype": self.resolved_dtype_name,
            "used_fallback": bool(self.used_fallback),
            "fallback_reason": self.fallback_reason,
        }


def resolve_torch_runtime(
    *,
    device: str = "auto",
    dtype_name: str | None = None,
    allow_cpu_fallback: bool = False,
) -> TorchRuntimeResolution:
    """Resolve torch device and dtype with explicit fallback reporting."""

    requested = _norm_device_name(device)
    requested_dtype = str(dtype_name).strip().lower() if dtype_name else None

    candidate = best_torch_device() if requested == "auto" else requested
    candidate = _norm_device_name(candidate)

    ok, reason = _device_available(candidate)
    used_fallback = False
    fallback_reason = None
    resolved = candidate

    if not ok:
        if not allow_cpu_fallback:
            raise RuntimeError(reason or f"Requested torch device '{candidate}' is unavailable.")
        resolved = "cpu"
        used_fallback = True
        fallback_reason = (
            f"Falling back to CPU because device '{candidate}' is unavailable. "
            f"{reason or ''}".strip()
        )

    dtype_choice = None if requested_dtype in {None, "", "auto"} else requested_dtype
    try:
        resolved_dtype = resolve_torch_dtype(
            device=str(resolved),
            dtype_name=dtype_choice or best_torch_dtype_name(device=str(resolved)),
        )
        resolved_dtype_name = str(resolved_dtype).replace("torch.", "")
    except Exception:
        resolved_dtype = None
        resolved_dtype_name = requested_dtype or "auto"

    return TorchRuntimeResolution(
        requested_device=requested,
        resolved_device=str(resolved),
        requested_dtype=requested_dtype,
        resolved_dtype_name=resolved_dtype_name,
        torch_dtype=resolved_dtype,
        used_fallback=used_fallback,
        fallback_reason=fallback_reason,
    )
