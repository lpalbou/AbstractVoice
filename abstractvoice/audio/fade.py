"""Small audio fade helpers (click / boundary-noise reduction).

Streaming TTS (especially when chunking text and synthesizing segments independently)
can produce audible clicks or short broadband "ticks" at chunk boundaries due to
waveform discontinuities.

This module provides lightweight, dependency-free edge fades to reduce those artifacts.
"""

from __future__ import annotations

import numpy as np


def apply_edge_fades(
    mono: np.ndarray,
    *,
    sample_rate: int,
    fade_ms: float = 5.0,
    fade_in: bool = True,
    fade_out: bool = True,
) -> np.ndarray:
    """Apply short linear fade-in/out to a mono float waveform (best-effort)."""
    x = np.asarray(mono, dtype=np.float32).reshape(-1)
    n = int(x.size)
    if n <= 0:
        return x

    try:
        sr = int(sample_rate)
    except Exception:
        sr = 0
    if sr <= 0:
        sr = 24000

    try:
        fm = float(fade_ms)
    except Exception:
        fm = 5.0
    if not (fm > 0.0):
        return x

    fade_n = int(round(float(sr) * float(fm) / 1000.0))
    if fade_n <= 0:
        return x

    # Avoid overlapping fades on very short arrays.
    if fade_in and fade_out and (2 * fade_n) > n:
        fade_n = max(0, n // 2)
    if fade_n <= 0:
        return x

    y = x.astype(np.float32, copy=True)
    # Use an endpoint-free ramp so n==1 yields [0.0] (guarantees zero at edges).
    ramp = (np.arange(fade_n, dtype=np.float32) / float(fade_n)).astype(np.float32, copy=False)

    if fade_in:
        y[:fade_n] *= ramp
    if fade_out:
        y[-fade_n:] *= ramp[::-1]
    return y


def ensure_headroom(mono: np.ndarray, *, headroom: float = 0.98) -> np.ndarray:
    """Scale down (only if needed) so peak amplitude stays under `headroom`."""
    x = np.asarray(mono, dtype=np.float32).reshape(-1)
    if x.size <= 0:
        return x
    try:
        hr = float(headroom)
    except Exception:
        hr = 0.98
    if not (0.0 < hr <= 1.0):
        hr = 0.98
    try:
        peak = float(np.max(np.abs(x)))
    except Exception:
        peak = 0.0
    if not (peak > 0.0):
        return x
    if peak <= hr:
        return x
    scale = hr / peak
    return (x * float(scale)).astype(np.float32, copy=False)

