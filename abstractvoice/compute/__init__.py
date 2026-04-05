"""Compute helpers (device selection, acceleration notes)."""

from .device import best_torch_device, best_faster_whisper_device
from .dtype import best_torch_dtype_name, resolve_torch_dtype

__all__ = [
    "best_torch_device",
    "best_faster_whisper_device",
    "best_torch_dtype_name",
    "resolve_torch_dtype",
]

