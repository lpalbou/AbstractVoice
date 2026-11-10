"""ONNX Runtime on an NVIDIA GPU: what the ONNX engines (Supertonic, Piper) share.

onnxruntime-gpu (the `onnx-cuda` extra) does not search the directories of
the nvidia-* wheels pip installs. Without `preload_cuda_libraries()` its CUDA
provider fails to load even with every library installed, and the session
runs on the CPU with only a log line (measured: Ubuntu 24.04, onnxruntime-gpu
1.31.1, with and without `import torch` first). A session's `get_providers()`
is therefore the only truth about where it runs.
"""

from __future__ import annotations

import contextlib
import ctypes
import functools
import io
import logging
import sys
import threading
from typing import Any

logger = logging.getLogger(__name__)

CUDA_PROVIDER = "CUDAExecutionProvider"

# Both distributions install the same `onnxruntime` package. A later (re)install of the CPU
# build, which piper-tts and faster-whisper require, writes its files over onnxruntime-gpu's;
# pip then sees onnxruntime-gpu as installed and does not rewrite them.
REINSTALL_CUDA_BUILD = "pip install --force-reinstall --no-deps onnxruntime-gpu"

# With the CUDA provider in the build but no device (CUDA_VISIBLE_DEVICES= , no driver), ONNX
# Runtime's Python wrapper prints an "EP Error" banner to STDOUT, then retries on the CPU. Stdout
# is a stdio server's protocol channel: the CUDA provider is requested only with a usable device.
NO_CUDA_DEVICE = "ONNX Runtime has the CUDA provider but no usable CUDA device"


# The preload runs once per process: it redirects stdout process-wide while it runs, so a
# print from another thread in that window would go to the log instead; once is the most.
_preload_lock = threading.Lock()
_preload_done = False
_preload_note: str | None = None  # what the one preload printed or raised (None: nothing)


def preload_cuda_libraries(ort: Any = None) -> str | None:
    """Load the CUDA libraries pip installed (nvidia-* wheels) before creating a CUDA session.

    Runs `preload_dlls` (ONNX Runtime 1.21+) at most once per process; every
    later call returns the first one's note without running it again.
    What it prints (the libraries it cannot find) is logged once, never
    written to stdout (an MCP stdio server's channel), and returned.
    """
    global _preload_done, _preload_note
    with _preload_lock:
        if _preload_done:
            return _preload_note
        if ort is None:
            try:
                import onnxruntime as ort  # type: ignore
            except ImportError:
                return None  # no ONNX Runtime: the engine reports that itself
        preload = getattr(ort, "preload_dlls", None)
        note = None
        if callable(preload):
            printed = io.StringIO()
            try:
                with contextlib.redirect_stdout(printed):
                    preload()
            except Exception as e:  # noqa: BLE001 - the session falls back to the CPU and says so
                note = f"ONNX Runtime could not preload its CUDA libraries: {e}"
            text = printed.getvalue().strip()
            if text:
                note = f"{note}; {text}" if note else f"ONNX Runtime CUDA libraries: {text}"
            if note:
                logger.warning("%s", note)
        _preload_done, _preload_note = True, note
        return note


@functools.lru_cache(maxsize=1)
def onnxruntime_gpu_installed() -> bool:
    """True when the onnxruntime-gpu distribution is installed (read once per process)."""
    from importlib import metadata

    try:
        metadata.version("onnxruntime-gpu")
    except metadata.PackageNotFoundError:
        return False
    return True


def _cudart_device_count() -> int | None:
    """`cudaGetDeviceCount` through the CUDA runtime library, or None when none loads."""
    names = (
        ("cudart64_13.dll", "cudart64_12.dll")
        if sys.platform == "win32"
        else ("libcudart.so.13", "libcudart.so.12", "libcudart.so")
    )
    for name in names:
        try:
            cudart = ctypes.CDLL(name)
        except OSError:
            continue
        count = ctypes.c_int(0)
        try:
            status = cudart.cudaGetDeviceCount(ctypes.byref(count))
        except Exception:  # noqa: BLE001 - a library without the symbol: try the next one
            continue
        return int(count.value) if status == 0 else 0  # cudaErrorNoDevice, no driver, ...
    return None


@functools.lru_cache(maxsize=1)
def usable_cuda_device() -> tuple[bool, str]:
    """Whether a CUDA device is usable in this process, and how that was decided (asked once).

    `torch.cuda.is_available()` when torch imports (it honours CUDA_VISIBLE_DEVICES and the
    driver; the gpu profiles carry torch), else the CUDA runtime's own device count. Cost: one
    torch import, once per process, on hosts whose ONNX Runtime has the CUDA provider.
    """
    try:
        import torch
    except Exception:  # noqa: BLE001 - not installed, or broken: ask the CUDA runtime instead
        torch = None
    if torch is not None:
        try:
            usable = bool(torch.cuda.is_available())
        except Exception as e:  # noqa: BLE001
            return False, f"torch.cuda.is_available() failed ({type(e).__name__}: {e})"
        return usable, f"torch.cuda.is_available() is {usable}"
    preload_cuda_libraries()  # the CUDA runtime of the nvidia-* wheels, when installed
    count = _cudart_device_count()
    if count is None:
        return False, "no CUDA runtime library loads to count the devices (torch is not installed)"
    return count > 0, f"cudaGetDeviceCount() reports {count} device(s)"


def no_cuda_device_reason() -> str | None:
    """Why the CUDA provider must not be requested here (None when a CUDA device is usable)."""
    usable, why = usable_cuda_device()
    return None if usable else f"{NO_CUDA_DEVICE}: {why}"
