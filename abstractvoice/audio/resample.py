from __future__ import annotations

import numpy as np


def linear_resample_mono(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Lightweight mono resampler (linear interpolation).

    Good enough for short prompts and avoids adding heavy DSP dependencies.
    """
    src_sr = int(src_sr)
    dst_sr = int(dst_sr)
    if src_sr <= 0 or dst_sr <= 0:
        return audio
    if src_sr == dst_sr:
        return audio
    if audio is None or len(audio) < 2:
        return audio

    ratio = float(dst_sr) / float(src_sr)
    new_len = max(1, int(round(len(audio) * ratio)))
    x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=True)
    x_new = np.linspace(0.0, 1.0, num=new_len, endpoint=True)
    return np.interp(x_new, x_old, audio).astype(np.float32)

