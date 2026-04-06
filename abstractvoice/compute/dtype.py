"""Torch dtype selection helpers.

We keep dtype selection centralized so new torch-based engines (TTS/cloning)
behave consistently across platforms.
"""

from __future__ import annotations

import os


def best_torch_dtype_name(*, device: str) -> str:
    """Return a recommended torch dtype name for `device`.

    Honors env var `ABSTRACTVOICE_TORCH_DTYPE` when set.
    """
    forced = (os.environ.get("ABSTRACTVOICE_TORCH_DTYPE") or "").strip().lower()
    if forced:
        # normalize common spellings
        if forced in ("fp16", "float16", "half"):
            return "float16"
        if forced in ("bf16", "bfloat16"):
            return "bfloat16"
        if forced in ("fp32", "float32"):
            return "float32"
        return forced

    dev = str(device or "").strip().lower()
    if dev == "cuda":
        # Prefer bf16 on modern NVIDIA GPUs, but allow fallback.
        return "bfloat16"
    if dev == "mps":
        return "float16"
    if dev == "xpu":
        # Conservative; engine can override.
        return "float16"
    return "float32"


def resolve_torch_dtype(*, device: str, dtype_name: str | None = None):
    """Resolve a torch dtype object for a device.

    This imports torch lazily so importing AbstractVoice stays lightweight.
    """
    try:
        import torch
    except Exception as e:  # pragma: no cover
        raise RuntimeError("torch is required to resolve a torch dtype") from e

    name = (dtype_name or best_torch_dtype_name(device=device)).strip().lower()
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype_name: {dtype_name}")

    dt = mapping[name]

    # CUDA bf16 support varies; fall back to fp16 if needed.
    if str(device).strip().lower() == "cuda" and dt == torch.bfloat16:
        try:
            if hasattr(torch.cuda, "is_bf16_supported") and not torch.cuda.is_bf16_supported():
                return torch.float16
        except Exception:
            # Best-effort; keep requested dtype.
            return dt

    return dt

