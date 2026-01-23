"""Compute helpers (device selection, acceleration notes)."""

from .device import best_torch_device, best_faster_whisper_device

__all__ = ["best_torch_device", "best_faster_whisper_device"]

