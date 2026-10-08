"""Device selection helpers.

We have multiple compute backends in this project:
- torch models (cloning): can use CUDA/MPS/XPU/CPU depending on local setup.
- faster-whisper / CTranslate2 (STT): CUDA or CPU (CTranslate2 has no Apple GPU backend;
  mlx-whisper is the Apple GPU engine).

Design goal: choose the best available device by default, while still allowing
explicit overrides in higher-level APIs.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Optional


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


# CTranslate2 compute types, best first, per device. CTranslate2 reports what a GPU supports
# (`get_supported_compute_types("cuda")`): float16 needs compute capability >= 5.3, the int8
# kernels >= 6.1, so an older card falls through to the next one it supports.
CUDA_COMPUTE_PREFERENCE = ("int8_float16", "float16", "int8", "float32")
CPU_COMPUTE_PREFERENCE = ("int8", "int8_float32", "float32")

WHISPER_DEVICE_ENV = "ABSTRACTVOICE_WHISPER_DEVICE"


@dataclass(frozen=True)
class WhisperDevice:
    """Where faster-whisper (CTranslate2) runs, and why (`resolve_faster_whisper_device`).

    `device` is "cuda" or "cpu"; `compute_type` the CTranslate2 type chosen for it; `reason`
    a plain sentence whenever Whisper does NOT run on a GPU (None on CUDA); `requested` the
    device that was asked for ("auto", or the value of ABSTRACTVOICE_WHISPER_DEVICE);
    `refused` True when an explicit request could not be honoured.
    """

    device: str
    compute_type: str
    reason: Optional[str] = None
    requested: str = "auto"
    refused: bool = False

    def to_dict(self) -> dict:
        return {
            "device": self.device,
            "compute_type": self.compute_type,
            "reason": self.reason,
            "requested": self.requested,
            "refused": self.refused,
        }


def _pick_compute_type(supported: Any, preference: tuple) -> Optional[str]:
    values = {str(v) for v in (supported or ())}
    for candidate in preference:
        if candidate in values:
            return candidate
    return None


def _cpu_compute_type(ct2: Any) -> str:
    try:
        picked = _pick_compute_type(ct2.get_supported_compute_types("cpu"), CPU_COMPUTE_PREFERENCE)
    except Exception:
        picked = None
    return picked or "int8"


def _import_ctranslate2() -> Any:
    if sys.platform == "win32" or sys.platform.startswith("linux"):
        from .windows_cuda import prepare_windows_cuda_dlls

        prepare_windows_cuda_dlls()
    import ctranslate2  # type: ignore

    return ctranslate2


def _cuda_unusable_reason(ct2: Any) -> Optional[str]:
    """None when CTranslate2 can run Whisper on CUDA here, else the sentence saying why not."""

    try:
        count = int(ct2.get_cuda_device_count())
    except Exception as exc:  # builds without CUDA raise here
        return f"this CTranslate2 build has no CUDA support ({exc})"
    if count <= 0:
        if sys.platform == "darwin":
            return (
                "CTranslate2 (faster-whisper) has no Apple GPU backend, so Whisper runs on the processor; "
                "mlx-whisper runs it on the Apple GPU"
            )
        return "CTranslate2 sees no CUDA GPU"
    from .windows_cuda import cublas12_available, cudnn9_available, CUBLAS12_DLL, CUBLAS12_SO, CUDNN9_DLL, CUDNN9_SO

    if not cublas12_available():
        name = CUBLAS12_DLL if sys.platform == "win32" else CUBLAS12_SO
        return (
            f"a CUDA GPU was found, but {name} (CUDA 12 cuBLAS, needed by CTranslate2) does not load; "
            'install AbstractVoice\'s gpu extra: pip install "abstractvoice[gpu]"'
        )
    if not cudnn9_available():
        name = CUDNN9_DLL if sys.platform == "win32" else CUDNN9_SO
        return (
            f"a CUDA GPU was found, but {name} (cuDNN 9 for CUDA 12, needed by the Whisper encoder on the GPU) "
            'does not load; install AbstractVoice\'s gpu extra: pip install "abstractvoice[gpu]"'
        )
    return None


def resolve_faster_whisper_device(requested: Optional[str] = None) -> WhisperDevice:
    """Choose faster-whisper's device and compute type, with the reason when it is not a GPU.

    `requested` ("auto", "cpu", "cuda"; None = the ABSTRACTVOICE_WHISPER_DEVICE setting, else
    "auto"). "auto" picks CUDA when CTranslate2 sees a CUDA GPU AND the CUDA 12 cuBLAS and
    cuDNN 9 it loads at the first GPU run are loadable; otherwise the CPU, with the reason.
    An explicit "cuda" that cannot be honoured is REFUSED: the CPU runs, `refused` is True and
    the reason says so (never a crash at the first transcription). Never imports torch.
    """

    source_env = requested is None
    raw = (os.environ.get(WHISPER_DEVICE_ENV) or "") if source_env else str(requested or "")
    want = raw.strip().lower() or "auto"
    if want.startswith("cuda"):
        want = "cuda"
    label = f"{WHISPER_DEVICE_ENV}={raw.strip()}" if (source_env and raw.strip()) else f"device {want!r}"

    try:
        ct2 = _import_ctranslate2()
    except Exception as exc:
        reason = f"CTranslate2 (faster-whisper) is not importable ({exc})"
        return WhisperDevice("cpu", "int8", reason, want, refused=want == "cuda")

    cpu_type = _cpu_compute_type(ct2)
    if want == "cpu":
        return WhisperDevice("cpu", cpu_type, f"the processor was requested ({label})", want)
    if want not in {"auto", "cuda"}:
        return WhisperDevice(
            "cpu", cpu_type, f"{label} is not a faster-whisper device (cpu, cuda or auto); using the processor",
            want, refused=True,
        )

    why_not = _cuda_unusable_reason(ct2)
    if why_not is None:
        try:
            cuda_type = _pick_compute_type(ct2.get_supported_compute_types("cuda"), CUDA_COMPUTE_PREFERENCE)
        except Exception:
            cuda_type = None
        if cuda_type:
            return WhisperDevice("cuda", cuda_type, None, want)
        why_not = "the CUDA GPU supports none of the compute types faster-whisper uses"
    if want == "cuda":
        return WhisperDevice("cpu", cpu_type, f"{label} refused: {why_not}; running on the processor", want, refused=True)
    return WhisperDevice("cpu", cpu_type, why_not, want)


def best_faster_whisper_device() -> str:
    """Return best device for faster-whisper: cuda|cpu (`resolve_faster_whisper_device().device`).

    Honors env var `ABSTRACTVOICE_WHISPER_DEVICE` (a request: "cuda" without a usable CUDA GPU
    is refused, with a reason, rather than returned). faster-whisper has no Apple GPU backend.
    """

    return resolve_faster_whisper_device().device
