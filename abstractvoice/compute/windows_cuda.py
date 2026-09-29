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

Linux has the same gap: CTranslate2's Linux wheel ``dlopen``s ``libcublas.so.12`` by name, but the
NVIDIA wheels (``nvidia-cublas-cu12``, pulled by PyTorch's CUDA 12 build or by the extras above)
unpack to ``site-packages/nvidia/<lib>/lib``, which is not on the loader's search path. PyTorch
preloads its own copies when it is imported, so faster-whisper worked only when torch happened to
be imported first, and failed with "Library libcublas.so.12 is not found or cannot be loaded" at the
first GPU transcription otherwise (measured on a Linux + NVIDIA gpu install, framework backlog
0989). On Linux `prepare_windows_cuda_dlls()` therefore preloads the CUDA 12 runtime and cuBLAS
from those folders with ``RTLD_GLOBAL`` (the way torch does), so a later load by name resolves to
them, and `cublas12_available()` asks the loader whether ``libcublas.so.12`` loads.

Everything here is a no-op off Windows and Linux and never imports torch or CTranslate2.
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
CUBLAS12_SO = "libcublas.so.12"
# `site-packages/nvidia/<name>/bin` folders of the Windows CUDA 12 wheels.
NVIDIA_CU12_LIBS = ("cublas", "cuda_runtime", "cudnn")
# Linux: the CUDA 12 libraries CTranslate2 loads by name, in dependency order, and the
# `site-packages/nvidia/<name>/lib` folder each one lives in.
LINUX_CU12_PRELOAD = (
    ("cuda_runtime", "libcudart.so.12"),
    ("cublas", "libcublasLt.so.12"),
    ("cublas", "libcublas.so.12"),
)

_prepared: Optional[List[str]] = None
_handles: list = []
_cublas12: Optional[bool] = None


def _nvidia_roots() -> List[Path]:
    """The ``site-packages/nvidia`` folders of the installed NVIDIA CUDA wheels."""

    try:
        spec = importlib.util.find_spec("nvidia")
    except Exception:
        return []
    if spec is None:
        return []
    return [Path(root) for root in (spec.submodule_search_locations or [])]


def linux_cu12_libraries() -> List[Path]:
    """Existing CUDA 12 runtime/cuBLAS files from the NVIDIA wheels, in load order (Linux layout)."""

    out: List[Path] = []
    for name, filename in LINUX_CU12_PRELOAD:
        for root in _nvidia_roots():
            candidate = root / name / "lib" / filename
            if candidate.is_file():
                out.append(candidate)
                break
    return out


def nvidia_cu12_bin_dirs() -> List[Path]:
    """Existing ``nvidia/<lib>/bin`` folders from the installed NVIDIA CUDA wheels."""

    roots = _nvidia_roots()
    out: List[Path] = []
    for root in roots:
        for name in NVIDIA_CU12_LIBS:
            folder = Path(root) / name / "bin"
            if folder.is_dir() and folder not in out:
                out.append(folder)
    return out


def prepare_windows_cuda_dlls() -> List[str]:
    """Make the NVIDIA CUDA 12 wheels findable before CTranslate2 needs them (once per process).

    Windows: adds their ``bin`` folders to the DLL search. Linux: preloads their CUDA 12 runtime
    and cuBLAS with ``RTLD_GLOBAL``. Returns the folders added (Windows) or the files preloaded
    (Linux); empty elsewhere or when the wheels are not installed."""

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
    elif sys.platform.startswith("linux"):
        for library in linux_cu12_libraries():
            try:
                _handles.append(ctypes.CDLL(str(library), mode=getattr(ctypes, "RTLD_GLOBAL", 0)))
                added.append(str(library))
            except OSError as exc:
                logger.debug("could not preload %s: %s", library, exc)
    _prepared = added
    return list(added)


def _load_library(name: str) -> bool:
    try:
        # winmode=0: the legacy search order (includes PATH), as CTranslate2's own load by name.
        ctypes.WinDLL(name, winmode=0)  # type: ignore[attr-defined]
        return True
    except (OSError, AttributeError, TypeError):
        return False


def _load_shared_object(name: str) -> bool:
    try:
        # By name, as CTranslate2 loads it: resolves to a preloaded copy or the loader's search path.
        ctypes.CDLL(name)
        return True
    except OSError:
        return False


def cublas12_available() -> bool:
    """True when CUDA 12 cuBLAS (``cublas64_12.dll`` / ``libcublas.so.12``) loads in this process.

    Windows and Linux only; True elsewhere, where this check does not apply. Cached after the
    first probe."""

    global _cublas12
    if sys.platform == "win32":
        name, load = CUBLAS12_DLL, _load_library
    elif sys.platform.startswith("linux"):
        name, load = CUBLAS12_SO, _load_shared_object
    else:
        return True
    if _cublas12 is None:
        prepare_windows_cuda_dlls()
        _cublas12 = load(name)
        if not _cublas12:
            logger.warning(
                "faster-whisper runs on the CPU: CUDA was found but %s (CUDA 12 cuBLAS, needed by "
                "CTranslate2) does not load. Install AbstractVoice's gpu extra (or the "
                "AbstractFramework installer's gpu setting), which adds it.",
                name,
            )
    return bool(_cublas12)


# The Windows-only name of `cublas12_available`, kept for callers of 0.13.x.
windows_cublas12_available = cublas12_available
