# Derived from mlx-audio 0.6.1 (https://github.com/Blaizzy/mlx-audio, MIT, Copyright (c) 2025
# Prince Canuma and contributors), file mlx_audio/tts/models/qwen3_tts/speaker_encoder.py, plus
# the ``mel_spectrogram`` front-end of mlx_audio/tts/models/qwen3_tts/qwen3_tts.py.
# Local modifications, listed exhaustively:
#   1. The mel front-end is NumPy (``mel_spectrogram``): reference clips are seconds long and the
#      Slaney filterbank is built here (``slaney_mel_filters``) instead of through mlx-audio's DSP
#      module, so the MLX path needs neither librosa nor transformers.
#   2. ``sanitize`` is replaced by ``weights.py`` (Conv1d weights transposed there).
#   3. Module attribute names mirror the PyTorch state dict (``speaker_encoder.*``).
"""ECAPA-TDNN speaker encoder (Base checkpoints) and its mel-spectrogram front-end."""

from __future__ import annotations

import math
from typing import List

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .config import Qwen3TTSSpeakerEncoderConfig

# The talker's speaker-embedding front-end, fixed by upstream (`extract_speaker_embedding`).
MEL_N_FFT = 1024
MEL_NUM_MELS = 128
MEL_SAMPLE_RATE = 24000
MEL_HOP = 256
MEL_WIN = 1024
MEL_FMIN = 0.0
MEL_FMAX = 12000.0


# ------------------------------------------------------------------ mel front-end (NumPy)


def _hz_to_mel_slaney(freq: np.ndarray) -> np.ndarray:
    f_sp = 200.0 / 3
    mels = freq / f_sp
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = math.log(6.4) / 27.0
    log_region = freq >= min_log_hz
    mels = np.where(log_region, min_log_mel + np.log(np.maximum(freq, 1e-12) / min_log_hz) / logstep, mels)
    return mels


def _mel_to_hz_slaney(mels: np.ndarray) -> np.ndarray:
    f_sp = 200.0 / 3
    freqs = f_sp * mels
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = math.log(6.4) / 27.0
    log_region = mels >= min_log_mel
    return np.where(log_region, min_log_hz * np.exp(logstep * (mels - min_log_mel)), freqs)


def slaney_mel_filters(*, sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    """``librosa.filters.mel`` with librosa defaults (Slaney scale, Slaney area norm): (n_mels, 1 + n_fft//2)."""
    fftfreqs = np.linspace(0.0, float(sr) / 2.0, 1 + n_fft // 2, dtype=np.float64)
    mel_pts = np.linspace(
        float(_hz_to_mel_slaney(np.asarray([fmin], dtype=np.float64))[0]),
        float(_hz_to_mel_slaney(np.asarray([fmax], dtype=np.float64))[0]),
        n_mels + 2,
        dtype=np.float64,
    )
    mel_f = _mel_to_hz_slaney(mel_pts)
    fdiff = np.diff(mel_f)
    ramps = mel_f[:, None] - fftfreqs[None, :]
    lower = -ramps[:-2] / fdiff[:-1, None]
    upper = ramps[2:] / fdiff[1:, None]
    weights = np.maximum(0.0, np.minimum(lower, upper))
    enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
    weights *= enorm[:, None]
    return weights.astype(np.float32)


def mel_spectrogram(audio: np.ndarray) -> np.ndarray:
    """Log-mel features as the PyTorch model computes them: (frames, 128), float32.

    Mirrors ``modeling_qwen3_tts.mel_spectrogram``: reflect padding of (n_fft - hop) // 2 on
    both sides, a periodic Hann window, ``center=False`` STFT, magnitude ``sqrt(re² + im² + 1e-9)``,
    Slaney mel filterbank, ``log(clamp(x, 1e-5))``.
    """
    y = np.asarray(audio, dtype=np.float32).reshape(-1)
    pad = (MEL_N_FFT - MEL_HOP) // 2
    if y.size <= pad:
        raise ValueError("reference audio is too short for a mel spectrogram")
    y = np.concatenate([y[1 : pad + 1][::-1], y, y[-(pad + 1) : -1][::-1]])
    n_frames = 1 + (y.size - MEL_N_FFT) // MEL_HOP
    if n_frames <= 0:
        raise ValueError("reference audio is too short for a mel spectrogram")
    idx = np.arange(MEL_N_FFT)[None, :] + (np.arange(n_frames) * MEL_HOP)[:, None]
    frames = y[idx]
    # torch.hann_window(win_size) is periodic: 0.5 - 0.5 cos(2 pi n / N).
    window = (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(MEL_WIN) / MEL_WIN)).astype(np.float32)
    spec = np.fft.rfft(frames * window, n=MEL_N_FFT, axis=-1)
    mag = np.sqrt(spec.real**2 + spec.imag**2 + 1e-9).astype(np.float32)
    mel_basis = slaney_mel_filters(sr=MEL_SAMPLE_RATE, n_fft=MEL_N_FFT, n_mels=MEL_NUM_MELS, fmin=MEL_FMIN, fmax=MEL_FMAX)
    mel = mag @ mel_basis.T
    return np.log(np.maximum(mel, 1e-5)).astype(np.float32)


# ------------------------------------------------------------------ ECAPA-TDNN (MLX)


def _reflect_pad_time(x: mx.array, pad: int) -> mx.array:
    """Reflect padding on the time axis of an NLC tensor."""
    if pad <= 0:
        return x
    left = x[:, 1 : pad + 1, :][:, ::-1, :]
    right = x[:, -(pad + 1) : -1, :][:, ::-1, :]
    return mx.concatenate([left, x, right], axis=1)


class TimeDelayNetBlock(nn.Module):
    """Conv1d (reflect "same" padding) + ReLU. Works in NCL like the PyTorch module."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.pad = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=dilation)

    def __call__(self, x: mx.array) -> mx.array:
        x = mx.transpose(x, (0, 2, 1))
        x = _reflect_pad_time(x, self.pad)
        x = self.conv(x)
        return nn.relu(mx.transpose(x, (0, 2, 1)))


class Res2NetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, scale: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.scale = int(scale)
        in_channel = in_channels // scale
        hidden_channel = out_channels // scale
        self.blocks = [
            TimeDelayNetBlock(in_channel, hidden_channel, kernel_size=kernel_size, dilation=dilation)
            for _ in range(scale - 1)
        ]

    def __call__(self, x: mx.array) -> mx.array:
        chunks = mx.split(x, self.scale, axis=1)
        outputs = []
        part = None
        for i, chunk in enumerate(chunks):
            if i == 0:
                part = chunk
            elif i == 1:
                part = self.blocks[i - 1](chunk)
            else:
                part = self.blocks[i - 1](chunk + part)
            outputs.append(part)
        return mx.concatenate(outputs, axis=1)


class SqueezeExcitationBlock(nn.Module):
    def __init__(self, in_channels: int, se_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, se_channels, kernel_size=1)
        self.conv2 = nn.Conv1d(se_channels, out_channels, kernel_size=1)

    def __call__(self, x: mx.array) -> mx.array:
        pooled = mx.mean(x, axis=2, keepdims=True)  # [B, C, 1]
        se = mx.transpose(pooled, (0, 2, 1))
        se = nn.relu(self.conv1(se))
        se = mx.sigmoid(self.conv2(se))
        return x * mx.transpose(se, (0, 2, 1))


class SqueezeExcitationRes2NetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        res2net_scale: int,
        se_channels: int,
        kernel_size: int,
        dilation: int,
    ) -> None:
        super().__init__()
        self.tdnn1 = TimeDelayNetBlock(in_channels, out_channels, kernel_size=1, dilation=1)
        self.res2net_block = Res2NetBlock(out_channels, out_channels, res2net_scale, kernel_size, dilation)
        self.tdnn2 = TimeDelayNetBlock(out_channels, out_channels, kernel_size=1, dilation=1)
        self.se_block = SqueezeExcitationBlock(out_channels, se_channels, out_channels)

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        x = self.tdnn1(x)
        x = self.res2net_block(x)
        x = self.tdnn2(x)
        x = self.se_block(x)
        return x + residual


class AttentiveStatisticsPooling(nn.Module):
    def __init__(self, channels: int, attention_channels: int) -> None:
        super().__init__()
        self.eps = 1e-12
        self.tdnn = TimeDelayNetBlock(channels * 3, attention_channels, 1, 1)
        self.conv = nn.Conv1d(attention_channels, channels, kernel_size=1)

    def __call__(self, x: mx.array) -> mx.array:
        batch, channels, length = x.shape
        mean = mx.mean(x, axis=2, keepdims=True)
        std = mx.sqrt(mx.var(x, axis=2, keepdims=True) + self.eps)
        attention = mx.concatenate(
            [x, mx.broadcast_to(mean, (batch, channels, length)), mx.broadcast_to(std, (batch, channels, length))],
            axis=1,
        )
        attention = mx.tanh(self.tdnn(attention))
        attention = mx.transpose(self.conv(mx.transpose(attention, (0, 2, 1))), (0, 2, 1))
        attention = mx.softmax(attention, axis=2)
        mean = mx.sum(attention * x, axis=2, keepdims=True)
        var = mx.sum(attention * (x - mean) ** 2, axis=2, keepdims=True)
        std = mx.sqrt(mx.clip(var, self.eps, None))
        return mx.concatenate([mean, std], axis=1)


class Qwen3TTSSpeakerEncoder(nn.Module):
    """``speaker_encoder.*``: log-mel frames (B, T, 128) -> speaker embedding (B, enc_dim)."""

    def __init__(self, config: Qwen3TTSSpeakerEncoderConfig) -> None:
        super().__init__()
        self.config = config
        channels: List[int] = list(config.enc_channels)
        self.blocks = [
            TimeDelayNetBlock(config.mel_dim, channels[0], config.enc_kernel_sizes[0], config.enc_dilations[0])
        ]
        for i in range(1, len(channels) - 1):
            self.blocks.append(
                SqueezeExcitationRes2NetBlock(
                    channels[i - 1],
                    channels[i],
                    res2net_scale=config.enc_res2net_scale,
                    se_channels=config.enc_se_channels,
                    kernel_size=config.enc_kernel_sizes[i],
                    dilation=config.enc_dilations[i],
                )
            )
        self.mfa = TimeDelayNetBlock(channels[-1], channels[-1], config.enc_kernel_sizes[-1], config.enc_dilations[-1])
        self.asp = AttentiveStatisticsPooling(channels[-1], attention_channels=config.enc_attention_channels)
        self.fc = nn.Conv1d(channels[-1] * 2, config.enc_dim, kernel_size=1)

    def __call__(self, mels: mx.array) -> mx.array:
        x = mx.transpose(mels, (0, 2, 1))  # NCL
        hidden = []
        for layer in self.blocks:
            x = layer(x)
            hidden.append(x)
        x = self.mfa(mx.concatenate(hidden[1:], axis=1))
        x = self.asp(x)
        x = mx.transpose(self.fc(mx.transpose(x, (0, 2, 1))), (0, 2, 1))
        return mx.squeeze(x, axis=-1)
