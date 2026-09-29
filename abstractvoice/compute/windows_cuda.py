"""Windows: the CUDA 12 cuBLAS that faster-whisper (CTranslate2) needs, on any torch stack.

CTranslate2's Windows wheel is built against CUDA 12 and loads ``cublas64_12.dll`` when it first
runs on the GPU. The AbstractFramework installer may pair the gpu setting with PyTorch's CUDA 13
build (``torch\\lib`` then carries ``cublas64_13.dll``, not 12), and NVIDIA publishes no
Windows wheel for CUDA 13 cuBLAS. So AbstractVoice's `gpu` / `all-gpu` extras add the CUDA 12
wheels on Windows only (``nvidia-cublas-cu12``, ``nvidia-cuda-runtime-cu12``; they unpack to
``site-packages\\nvidia\\<lib>\\bin``), and this module puts those folders on the DLL search before
CTranslate2 needs them -- both ``os.add_dll_directory`` and the front of ``PATH``, since a plain
``LoadLibrary`` by name only searches ``PATH`` (framework backlog 0988).

`windows_cublas12_available()` then asks Windows itself whether ``cublas64_12.dll`` loads
(``ctypes``), so `best_faster_whisper_device()` picks CUDA only when the library is really there
and otherwise falls back to the CPU, instead of failing on the first transcription.

Everything here is a no-op off Windows and never imports torch or CTranslate2.
"""

from __future__ import annotations

import ctypes
import importlib.util
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

CUBLAS12_DLL = "cublas64_12.dll"
# `site-packages/nvidia/<name>/bin` folders of the Windows CUDA 12 wheels.
NVIDIA_CU12_LIBS = ("cublas", "cuda_runtime", "cudnn")

_prepared: Optional[List[str]] = None
_handles: list = []
_cublas12: Optional[bool] = None


def nvidia_cu12_bin_dirs() -> List[Path]:
    """Existing ``nvidia/<lib>/bin`` folders from the installed NVIDIA CUDA wheels."""

    try:
        spec = importlib.util.find_spec("nvidia")
    except Exception:
        return []
    if spec is None:
        return []
    roots = list(spec.submodule_search_locations or [])
    out: List[Path] = []
    for root in roots:
        for name in NVIDIA_CU12_LIBS:
            folder = Path(root) / name / "bin"
            if folder.is_dir() and folder not in out:
                out.append(folder)
    return out


def prepare_windows_cuda_dlls() -> List[str]:
    """Add the NVIDIA CUDA 12 wheel folders to the Windows DLL search (once per process).

    Returns the folders added; empty off Windows or when the wheels are not installed."""

    global _prepared
    if _prepared is not None:
        return list(_prepared)
    added: List[str] = []
    if sys.platform == "win32":
        for folder in nvidia_cu12_bin_dirs():
            text = str(folder)
            add = getattr(os, "add_dll_directory", None)
            if callable(add):
                try:
                    # Keep the handle alive: closing it removes the folder again.
                    _handles.append(add(text))
                except OSError:
                    pass
            entries = os.environ.get("PATH", "").split(os.pathsep)
            if text not in entries:
                os.environ["PATH"] = text + (os.pathsep + os.environ["PATH"] if os.environ.get("PATH") else "")
            added.append(text)
    _prepared = added
    return list(added)


def _load_library(name: str) -> bool:
    try:
        # winmode=0: the legacy search order (includes PATH), as CTranslate2's own load by name.
        ctypes.WinDLL(name, winmode=0)  # type: ignore[attr-defined]
        return True
    except (OSError, AttributeError, TypeError):
        return False


def windows_cublas12_available() -> bool:
    """True when ``cublas64_12.dll`` loads in this process (Windows only; True elsewhere,
    where this check does not apply). Cached after the first probe."""

    global _cublas12
    if sys.platform != "win32":
        return True
    if _cublas12 is None:
        prepare_windows_cuda_dlls()
        _cublas12 = _load_library(CUBLAS12_DLL)
        if not _cublas12:
            logger.warning(
                "faster-whisper runs on the CPU: CUDA was found but %s (CUDA 12 cuBLAS, needed by "
                "CTranslate2) does not load. Reinstall with the AbstractFramework installer's gpu "
                "setting, which adds it.",
                CUBLAS12_DLL,
            )
    return bool(_cublas12)
