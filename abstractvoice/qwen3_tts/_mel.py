"""librosa-compatible mel filterbank without librosa.

Upstream Qwen3-TTS calls ``librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax)``
with librosa defaults (Slaney mel scale, Slaney area normalization) to feed its
speaker encoder. transformers ships the identical construction as
``transformers.audio_utils.mel_filter_bank``; measured parity against librosa
on the model's exact parameters is 3.7e-9 in float32, ~1 ULP after upstream's
own ``.float()`` cast. ``triangularize_in_mel_space`` must stay False — that
variant diverges by 2.5.
"""

from __future__ import annotations

import numpy as np


def librosa_mel_fn(*, sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float | None) -> np.ndarray:
    """Drop-in for ``librosa.filters.mel`` as Qwen3-TTS parametrizes it."""
    from transformers.audio_utils import mel_filter_bank

    filters = mel_filter_bank(
        num_frequency_bins=1 + n_fft // 2,
        num_mel_filters=int(n_mels),
        min_frequency=float(fmin),
        max_frequency=float(fmax) if fmax is not None else float(sr) / 2.0,
        sampling_rate=int(sr),
        norm="slaney",
        mel_scale="slaney",
        triangularize_in_mel_space=False,
    )
    # librosa returns (n_mels, 1 + n_fft // 2); transformers returns the transpose.
    return np.asarray(filters, dtype=np.float32).T
