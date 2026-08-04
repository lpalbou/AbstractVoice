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


def sinc_resample_mono(audio: np.ndarray, src_sr: int, dst_sr: int, *, zeros: int = 24) -> np.ndarray:
    """Band-limited mono resampler (windowed-sinc, Kaiser window). numpy only.

    Linear interpolation does not band-limit: downsampling 48k->24k folds all
    content above the new Nyquist straight into the pass band, and a speaker
    encoder fed those aliases learns the wrong voice. Every reference-audio and
    codec-input path must use this one; `linear_resample_mono` stays for
    latency-tolerant playback-side conversions of already-band-limited audio.

    Polyphase evaluation of a Kaiser-windowed sinc low-passed at the smaller
    Nyquist. `zeros` is the number of sinc zero crossings per side (24 gives
    ~90 dB alias rejection; plenty ahead of a 16-bit noise floor).
    """
    src_sr = int(src_sr)
    dst_sr = int(dst_sr)
    if src_sr <= 0 or dst_sr <= 0 or src_sr == dst_sr:
        return np.asarray(audio, dtype=np.float32)
    x = np.asarray(audio, dtype=np.float64).reshape(-1)
    if x.size < 2:
        return x.astype(np.float32)

    from math import gcd

    g = gcd(src_sr, dst_sr)
    up, down = dst_sr // g, src_sr // g

    # Anti-alias/anti-image cutoff at the smaller of the two Nyquists.
    cutoff = 0.5 * min(1.0, up / down)
    half_width = zeros / (2.0 * cutoff)  # taps per side, in input samples
    filter_half = int(np.ceil(half_width * up))
    n = np.arange(-filter_half, filter_half + 1, dtype=np.float64) / up
    kernel = 2.0 * cutoff * np.sinc(2.0 * cutoff * n) * np.kaiser(n.size, 14.0)

    # True polyphase evaluation. Zero-stuffing the whole signal and convolving
    # (the naive route) is O(N * up * taps): 16 seconds and ~750 MB for ten
    # seconds of 44.1 kHz audio. Only every `up`-th kernel tap ever meets a
    # nonzero input sample, so evaluate output sample m directly:
    #
    #   y[m] = sum_i x[i] * kernel[c - i*up],  c = m*down + filter_half
    #
    # For fixed m only `taps = ceil(len(kernel)/up)` input samples contribute,
    # with kernel indices congruent to p = c mod up. Splitting the kernel into
    # `up` phase rows turns each output sample into one `taps`-long dot product.
    kernel *= up / kernel.sum()  # exact unity DC gain through the decimation
    taps = (kernel.size + up - 1) // up
    kernel_padded = np.concatenate([kernel, np.zeros(up)])
    # phase_table[p][j] pairs with x[i_hi - j]; reverse once so rows pair with
    # ascending-time windows.
    phase_table = np.stack([kernel_padded[p::up][:taps] for p in range(up)])[:, ::-1].copy()

    expected = int(round(x.size * dst_sr / src_sr))
    pad = taps
    x_padded = np.concatenate([np.zeros(pad), x, np.zeros(pad + taps)])

    out = np.empty(expected, dtype=np.float64)
    m = np.arange(expected, dtype=np.int64)
    center = m * down + filter_half
    i_hi = center // up
    starts = i_hi - taps + 1 + pad
    phases = (center % up).astype(np.int64)

    # Chunked gather keeps peak memory ~ chunk*taps floats regardless of length.
    chunk = max(1, min(expected, 65536))
    window_idx = np.arange(taps, dtype=np.int64)
    for begin in range(0, expected, chunk):
        end = min(begin + chunk, expected)
        gather = x_padded[starts[begin:end, None] + window_idx[None, :]]
        out[begin:end] = np.einsum("ij,ij->i", gather, phase_table[phases[begin:end]])

    return out.astype(np.float32)

