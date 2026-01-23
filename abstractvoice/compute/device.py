"""Device selection helpers.

We have multiple compute backends in this project:
- torch models (cloning): can use CUDA/MPS/XPU/CPU depending on local setup.
- faster-whisper / CTranslate2 (STT): CUDA or CPU (no MPS backend today).

Design goal: choose the best available device by default, while still allowing
explicit overrides in higher-level APIs.
"""

from __future__ import annotations

import os
import sys


def best_torch_device() -> str:
    """Return best torch device string: cuda|mps|xpu|cpu.

    Honors env var `ABSTRACTVOICE_TORCH_DEVICE` when set (e.g. "cpu", "mps", "cuda").
    """
    forced = (os.environ.get("ABSTRACTVOICE_TORCH_DEVICE") or "").strip().lower()
    if forced:
        return forced

    try:
        import torch

        # CUDA (NVIDIA, and often ROCm via the CUDA API surface in PyTorch builds)
        if torch.cuda.is_available():
            return "cuda"

        # Apple Silicon (preferred on macOS when available)
        if sys.platform == "darwin":
            try:
                if torch.backends.mps.is_available():
                    return "mps"
            except Exception:
                pass

        # Intel XPU
        try:
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                return "xpu"
        except Exception:
            pass

    except Exception:
        pass

    return "cpu"


def best_faster_whisper_device() -> str:
    """Return best device for faster-whisper: cuda|cpu.

    Honors env var `ABSTRACTVOICE_WHISPER_DEVICE`.
    Note: faster-whisper doesn't support MPS as a backend today.
    """
    forced = (os.environ.get("ABSTRACTVOICE_WHISPER_DEVICE") or "").strip().lower()
    if forced:
        return forced

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass

    return "cpu"

