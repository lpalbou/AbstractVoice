"""Time-stretch helpers (speed change without pitch change).

We prefer `librosa.effects.time_stretch` when available, but keep a lightweight
fallback (WSOLA) so `/speed` works even when `abstractvoice[audio-fx]` isn't
installed.

This is intentionally mono-only and tuned for speech.
"""

from __future__ import annotations

import math

import numpy as np


def wsola_time_stretch(
    audio: np.ndarray,
    *,
    rate: float,
    sr: int,
    window_s: float = 0.04,
    hop_s: float = 0.01,
    search_s: float = 0.02,
) -> np.ndarray:
    """WSOLA time-stretch (best-effort, speech-focused).

    `rate` matches librosa semantics:
    - `rate > 1.0` => faster (shorter output)
    - `rate < 1.0` => slower (longer output)
    """

    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return x

    try:
        r = float(rate)
    except Exception:
        return x
    if not (r > 0.0) or not math.isfinite(r) or r == 1.0:
        return x

    sr_i = int(sr) if int(sr) > 0 else 22050
    win = max(64, int(round(float(window_s) * float(sr_i))))
    hop_out = max(16, int(round(float(hop_s) * float(sr_i))))
    overlap = max(16, win - hop_out)
    hop_in = max(1, int(round(float(hop_out) * float(r))))
    search = max(0, int(round(float(search_s) * float(sr_i))))

    # Very short signals: WSOLA isn't stable; return original.
    if x.size < win + hop_out:
        return x

    # Pad so correlation windows never go out of bounds.
    pad = win + search + 2
    xp = np.pad(x, (pad, pad), mode="constant").astype(np.float32, copy=False)

    target_len = int(round(float(x.size) / float(r)))
    # Reserve enough space for overlap-add.
    yp = np.zeros((target_len + 2 * pad + win + 2,), dtype=np.float32)
    wp = np.zeros_like(yp)

    window = np.hanning(win).astype(np.float32)

    in_pos = pad
    out_pos = pad

    def _add(seg: np.ndarray, pos: int) -> None:
        yp[pos : pos + win] += seg * window
        wp[pos : pos + win] += window

    # Prime with the first frame.
    _add(xp[in_pos : in_pos + win], out_pos)
    in_pos += hop_in
    out_pos += hop_out

    # Main WSOLA loop.
    while (out_pos + win) < yp.size and (in_pos + win + search) < xp.size:
        # Reference overlap region is what we already synthesized at this position.
        ref = yp[out_pos : out_pos + overlap]
        ref_w = wp[out_pos : out_pos + overlap]
        ref = ref / (ref_w + 1e-6)
        ref = ref - float(ref.mean())
        ref_norm = float(np.linalg.norm(ref)) + 1e-9

        pred = int(in_pos)
        if search <= 0:
            best = pred
        else:
            start = max(pad, pred - search)
            end = min(xp.size - win - pad, pred + search)
            best = pred
            best_score = -1.0e30
            for cand in range(start, end + 1):
                seg = xp[cand : cand + overlap]
                seg = seg - float(seg.mean())
                seg_norm = float(np.linalg.norm(seg)) + 1e-9
                score = float(np.dot(ref, seg)) / (ref_norm * seg_norm)
                if score > best_score:
                    best_score = score
                    best = cand

        _add(xp[best : best + win], out_pos)
        out_pos += hop_out
        in_pos = best + hop_in

    # Normalize overlap-add weights.
    out = yp / np.maximum(wp, 1e-6)
    out = out[pad : pad + target_len]
    return np.asarray(out, dtype=np.float32).reshape(-1)

