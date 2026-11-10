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
import functools
import io
import logging
from typing import Any

logger = logging.getLogger(__name__)

CUDA_PROVIDER = "CUDAExecutionProvider"

# Both distributions install the same `onnxruntime` package. A later (re)install of the CPU
# build, which piper-tts and faster-whisper require, writes its files over onnxruntime-gpu's;
# pip then sees onnxruntime-gpu as installed and does not rewrite them.
REINSTALL_CUDA_BUILD = "pip install --force-reinstall --no-deps onnxruntime-gpu"


def preload_cuda_libraries(ort: Any = None) -> None:
    """Load the CUDA libraries pip installed (nvidia-* wheels) before creating a CUDA session.

    `preload_dlls` (ONNX Runtime 1.21+) prints what it cannot find: that is
    logged, never written to stdout (an MCP stdio server's channel).
    """
    if ort is None:
        try:
            import onnxruntime as ort  # type: ignore
        except ImportError:
            return  # no ONNX Runtime: the engine reports that itself
    preload = getattr(ort, "preload_dlls", None)
    if not callable(preload):
        return
    printed = io.StringIO()
    try:
        with contextlib.redirect_stdout(printed):
            preload()
    except Exception as e:  # noqa: BLE001 - the session falls back to the CPU and says so
        logger.warning("ONNX Runtime could not preload its CUDA libraries: %s", e)
    text = printed.getvalue().strip()
    if text:
        logger.warning("ONNX Runtime CUDA libraries: %s", text)


@functools.lru_cache(maxsize=1)
def onnxruntime_gpu_installed() -> bool:
    """True when the onnxruntime-gpu distribution is installed (read once per process)."""
    from importlib import metadata

    try:
        metadata.version("onnxruntime-gpu")
    except metadata.PackageNotFoundError:
        return False
    return True
