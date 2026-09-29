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
        # Be tolerant of common spellings like "cuda:0".
        if forced.startswith("cuda"):
            return "cuda"
        return forced

    # IMPORTANT:
    # faster-whisper is backed by CTranslate2 and can be CUDA-enabled even when
    # PyTorch is not installed. Therefore, CUDA detection must not rely on torch.
    if _ctranslate2_sees_cuda():
        # Windows: CTranslate2 loads CUDA 12 cuBLAS lazily, at the first GPU run. A torch CUDA 13
        # stack does not carry it, so pick CUDA only when cublas64_12.dll really loads
        # (framework backlog 0988); otherwise the CPU works where CUDA would fail mid-call.
        from .windows_cuda import windows_cublas12_available

        if windows_cublas12_available():
            return "cuda"
    return "cpu"


def _ctranslate2_sees_cuda() -> bool:
    if sys.platform == "win32":
        from .windows_cuda import prepare_windows_cuda_dlls

        prepare_windows_cuda_dlls()
    try:
        import ctranslate2  # type: ignore

        # Preferred: explicit device count (available on modern ctranslate2).
        if hasattr(ctranslate2, "get_cuda_device_count"):
            try:
                n = int(ctranslate2.get_cuda_device_count())  # type: ignore[attr-defined]
                if n > 0:
                    return True
            except Exception:
                # If the package wasn't compiled with CUDA support, some builds raise.
                pass

        # Fallback: probe supported compute types on CUDA device 0.
        if hasattr(ctranslate2, "get_supported_compute_types"):
            try:
                types = ctranslate2.get_supported_compute_types("cuda", 0)  # type: ignore[attr-defined]
                if types:
                    return True
            except Exception:
                pass
    except Exception:
        # If ctranslate2 isn't importable, keep a conservative fallback.
        pass
    return False

