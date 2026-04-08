"""TTS audio delivery mode normalization (buffered vs streamed).

This is an engine-agnostic *delivery* concept:
- buffered: synthesize full audio first (one "WAV-like" payload), then play/return it
- streamed: synthesize and deliver audio incrementally in chunks when possible

Not all backends can truly stream generation; when unavailable, higher-level code
may fall back to buffered delivery while keeping API semantics stable.
"""

from __future__ import annotations

from typing import Literal, Union

AudioDeliveryMode = Literal["buffered", "streamed"]


def normalize_audio_delivery_mode(value: Union[str, bool, None]) -> AudioDeliveryMode:
    """Normalize user input into an `AudioDeliveryMode`.

    Accepted values (case-insensitive):
    - buffered: "buffered", "full", "wav", "file", "off", "false", "0"
    - streamed: "streamed", "stream", "chunks", "chunked", "on", "true", "1"
    - bool: True -> streamed, False -> buffered
    """
    if isinstance(value, bool):
        return "streamed" if value else "buffered"

    s = str(value or "").strip().lower()
    if not s:
        raise ValueError("mode must be non-empty")

    if s in ("buffered", "full", "wav", "file", "off", "false", "0"):
        return "buffered"
    if s in ("streamed", "stream", "chunks", "chunked", "on", "true", "1"):
        return "streamed"

    raise ValueError(f"Unsupported delivery mode: {value!r} (expected buffered|streamed)")

