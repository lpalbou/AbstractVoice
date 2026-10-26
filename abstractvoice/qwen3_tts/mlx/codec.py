# Derived from mlx-audio 0.6.1 (https://github.com/Blaizzy/mlx-audio, MIT, Copyright (c) 2025
# Prince Canuma and contributors), file mlx_audio/tts/models/qwen3_tts/speech_tokenizer.py, and
# from the Kyutai Mimi modules it builds the codec ENCODER on (mlx_audio/codec/models/mimi/*,
# MIT, Copyright (c) Kyutai). Local modifications, listed exhaustively:
#   1. Module attribute names mirror the PyTorch state dict (``decoder.*`` / ``encoder.*`` of
#      ``speech_tokenizer/model.safetensors``) so ``weights.py`` loads by name, strictly, with a
#      single rename (``_codebook`` -> ``codebook``: MLX hides underscore attributes from
#      ``parameters()``). The codebooks keep upstream's ``cluster_usage``/``embedding_sum``
#      parameters and derive the embedding table once after load (``finalize``).
#   2. The decoder transformer applies upstream's SLIDING-WINDOW causal attention (config
#      ``sliding_window``, 72 frames): mlx-audio attended over the whole history.
#   3. The encoder is laid out like Hugging Face's ``MimiEncoder``/``MimiTransformerModel``
#      (flat ``layers`` list with parameter-free ELUs, exact GELU, LayerNorm with bias,
#      sliding-window 250 attention) instead of Kyutai's Seanet/ProjectedTransformer classes, so no
#      key remapping is needed; only the first ``encoder_valid_num_quantizers`` codebooks are used.
#   4. Streaming decode is the ONLY decode path (``Qwen3TTSCodecDecoder.step``): it is the exact
#      causal computation, fed either one chunk at a time (streamed speech) or all frames at once
#      (buffered speech). mlx-audio's ``chunked_decode`` (300-frame windows with 25 frames of
#      context, an approximation) is not kept.
"""Qwen3-TTS 12Hz speech codec: streaming decoder (codes -> waveform) and encoder (waveform -> codes)."""

from __future__ import annotations

import math
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn

from ._cache import KVCache
from .config import Qwen3TTSCodecConfig, Qwen3TTSCodecDecoderConfig, Qwen3TTSCodecEncoderConfig


def _sliding_causal_mask(q_len: int, k_len: int, offset: int, window: Optional[int], dtype) -> Optional[mx.array]:
    """Additive mask: query at absolute position q sees key k iff k <= q and (no window or q - k < window).

    Keys cover absolute positions ``k_len - 1 - ... `` ending at ``offset + q_len - 1``.
    """
    if q_len == 1 and window is None:
        return None
    q_pos = mx.arange(offset, offset + q_len)[:, None]
    k_pos = mx.arange(offset + q_len - k_len, offset + q_len)[None, :]
    allowed = k_pos <= q_pos
    if window is not None:
        allowed = allowed & ((q_pos - k_pos) < window)
    return mx.where(allowed, mx.array(0.0, dtype=dtype), mx.array(-mx.inf, dtype=dtype))


def _rotate_half(x: mx.array) -> mx.array:
    half = x.shape[-1] // 2
    return mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def _rope_cos_sin(positions: mx.array, head_dim: int, theta: float, dtype):
    inv_freq = 1.0 / (theta ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim))
    freqs = positions.astype(mx.float32)[:, None] * inv_freq[None, :]
    emb = mx.concatenate([freqs, freqs], axis=-1)
    return mx.cos(emb).astype(dtype)[None, None], mx.sin(emb).astype(dtype)[None, None]


# ============================================================================ decoder


class CausalConv1d(nn.Module):
    """``Qwen3TTSTokenizerV2CausalConvNet``: left-padded conv in NLC, with a streaming buffer."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, *, stride: int = 1, dilation: int = 1, groups: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=0, dilation=dilation, groups=groups)
        effective = (kernel_size - 1) * dilation + 1
        self.padding = effective - stride
        self._buffer: Optional[mx.array] = None

    def reset_state(self) -> None:
        self._buffer = None

    def step(self, x: mx.array) -> mx.array:
        if self.padding > 0:
            if self._buffer is None:
                x = mx.pad(x, [(0, 0), (self.padding, 0), (0, 0)])
            else:
                x = mx.concatenate([self._buffer, x], axis=1)
            self._buffer = x[:, -self.padding :, :]
        return self.conv(x)


class CausalTransposeConv1d(nn.Module):
    """``Qwen3TTSTokenizerV2CausalTransConvNet``: transposed conv, right-trimmed, with overlap-add state."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, padding=0)
        self.trim_right = kernel_size - stride
        self._overflow: Optional[mx.array] = None

    def reset_state(self) -> None:
        self._overflow = None

    def step(self, x: mx.array) -> mx.array:
        y = self.conv(x)
        if self._overflow is not None:
            n = self._overflow.shape[1]
            y = mx.concatenate([y[:, :n, :] + self._overflow, y[:, n:, :]], axis=1)
        if self.trim_right > 0:
            tail = y[:, -self.trim_right :, :]
            if "bias" in self.conv:
                tail = tail - self.conv.bias  # the next chunk's head carries the bias already
            self._overflow = tail
            y = y[:, : -self.trim_right, :]
        return y


class SnakeBeta(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.alpha = mx.zeros((channels,))
        self.beta = mx.zeros((channels,))

    def __call__(self, x: mx.array) -> mx.array:
        alpha = mx.exp(self.alpha)
        beta = mx.exp(self.beta)
        return x + (1.0 / (beta + 1e-9)) * mx.square(mx.sin(x * alpha))


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dwconv = CausalConv1d(dim, dim, 7, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = mx.ones((dim,)) * 1e-6

    def reset_state(self) -> None:
        self.dwconv.reset_state()

    def step(self, x: mx.array) -> mx.array:
        residual = x
        x = self.dwconv.step(x)
        x = self.pwconv2(nn.gelu(self.pwconv1(self.norm(x))))
        return residual + self.gamma * x


class DecoderResidualUnit(nn.Module):
    def __init__(self, dim: int, dilation: int) -> None:
        super().__init__()
        self.act1 = SnakeBeta(dim)
        self.conv1 = CausalConv1d(dim, dim, 7, dilation=dilation)
        self.act2 = SnakeBeta(dim)
        self.conv2 = CausalConv1d(dim, dim, 1)

    def reset_state(self) -> None:
        self.conv1.reset_state()
        self.conv2.reset_state()

    def step(self, x: mx.array) -> mx.array:
        residual = x
        x = self.conv1.step(self.act1(x))
        x = self.conv2.step(self.act2(x))
        return x + residual


class DecoderBlock(nn.Module):
    """``decoder.decoder.{1..4}``: Snake, upsampling transposed conv, three residual units."""

    def __init__(self, config: Qwen3TTSCodecDecoderConfig, layer_idx: int) -> None:
        super().__init__()
        in_dim = config.decoder_dim // (2**layer_idx)
        out_dim = config.decoder_dim // (2 ** (layer_idx + 1))
        rate = config.upsample_rates[layer_idx]
        self.block = [
            SnakeBeta(in_dim),
            CausalTransposeConv1d(in_dim, out_dim, 2 * rate, rate),
            DecoderResidualUnit(out_dim, 1),
            DecoderResidualUnit(out_dim, 3),
            DecoderResidualUnit(out_dim, 9),
        ]

    def reset_state(self) -> None:
        for m in self.block[1:]:
            m.reset_state()

    def step(self, x: mx.array) -> mx.array:
        x = self.block[0](x)
        x = self.block[1].step(x)
        for unit in self.block[2:]:
            x = unit.step(x)
        return x


class _DecoderRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, self.eps)


class _LayerScale(nn.Module):
    def __init__(self, dim: int, initial_scale: float) -> None:
        super().__init__()
        self.scale = mx.ones((dim,)) * initial_scale

    def __call__(self, x: mx.array) -> mx.array:
        return self.scale * x


class _DecoderAttention(nn.Module):
    def __init__(self, config: Qwen3TTSCodecDecoderConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array, mask, cache: Optional[KVCache]) -> mx.array:
        B, T, _ = x.shape
        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, T, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, T, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(B, T, -1))


class _DecoderMLP(nn.Module):
    def __init__(self, config: Qwen3TTSCodecDecoderConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class _DecoderTransformerLayer(nn.Module):
    def __init__(self, config: Qwen3TTSCodecDecoderConfig) -> None:
        super().__init__()
        self.self_attn = _DecoderAttention(config)
        self.mlp = _DecoderMLP(config)
        self.input_layernorm = _DecoderRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = _DecoderRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn_layer_scale = _LayerScale(config.hidden_size, config.layer_scale_initial_scale)
        self.mlp_layer_scale = _LayerScale(config.hidden_size, config.layer_scale_initial_scale)

    def __call__(self, x: mx.array, cos, sin, mask, cache: Optional[KVCache]) -> mx.array:
        x = x + self.self_attn_layer_scale(self.self_attn(self.input_layernorm(x), cos, sin, mask, cache))
        return x + self.mlp_layer_scale(self.mlp(self.post_attention_layernorm(x)))


class DecoderTransformer(nn.Module):
    """``decoder.pre_transformer``: latent -> hidden transformer -> latent, sliding-window causal."""

    def __init__(self, config: Qwen3TTSCodecDecoderConfig) -> None:
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.latent_dim, config.hidden_size)
        self.layers = [_DecoderTransformerLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = _DecoderRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.output_proj = nn.Linear(config.hidden_size, config.latent_dim)
        self._cache: Optional[List[KVCache]] = None
        self._position = 0  # absolute frame index of the next input (RoPE uses absolute positions)

    def reset_state(self) -> None:
        self._cache = None
        self._position = 0

    def step(self, x: mx.array) -> mx.array:
        if self._cache is None:
            self._cache = [KVCache() for _ in self.layers]
        pos = self._position
        T = x.shape[1]
        k_len = self._cache[0].offset + T
        x = self.input_proj(x)
        cos, sin = _rope_cos_sin(mx.arange(pos, pos + T), self.config.head_dim, self.config.rope_theta, x.dtype)
        mask = _sliding_causal_mask(T, k_len, pos, self.config.sliding_window, x.dtype)
        for layer, cache in zip(self.layers, self._cache):
            x = layer(x, cos, sin, mask, cache)
        x = self.norm(x)
        x = self.output_proj(x)
        self._position = pos + T
        # Keys older than the window can never be attended again: keep the cache bounded. The
        # keys stay rotated at their absolute positions; only the mask/RoPE positions matter.
        window = self.config.sliding_window
        if window is not None:
            for cache in self._cache:
                if cache.offset > window:
                    cache.keys = cache.keys[..., cache.offset - window : cache.offset, :]
                    cache.values = cache.values[..., cache.offset - window : cache.offset, :]
                    cache.offset = window
        return x


class EuclideanCodebook(nn.Module):
    """Upstream's EMA codebook: the table is ``embedding_sum / max(cluster_usage, eps)``."""

    def __init__(self, dim: int, codebook_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.cluster_usage = mx.ones((codebook_size,))
        self.embedding_sum = mx.zeros((codebook_size, dim))
        self._embedding: Optional[mx.array] = None

    def finalize(self) -> None:
        self._embedding = self.embedding_sum / mx.maximum(self.cluster_usage, self.eps)[:, None]
        mx.eval(self._embedding)

    @property
    def embedding(self) -> mx.array:
        if self._embedding is None:
            self.finalize()
        return self._embedding

    def decode(self, codes: mx.array) -> mx.array:
        return self.embedding[codes]

    def encode(self, x: mx.array) -> mx.array:
        """Nearest centroid of each vector in ``x`` (..., dim) -> (...,) indices."""
        shape = x.shape[:-1]
        flat = x.reshape(-1, x.shape[-1]).astype(mx.float32)
        table = self.embedding.astype(mx.float32)
        dists = mx.sum(flat * flat, axis=-1, keepdims=True) - 2.0 * (flat @ table.T) + mx.sum(table * table, axis=-1)[None, :]
        return mx.argmin(dists, axis=-1).reshape(shape)


class VectorQuantization(nn.Module):
    def __init__(self, dim: int, codebook_size: int) -> None:
        super().__init__()
        self.codebook = EuclideanCodebook(dim, codebook_size)

    def decode(self, codes: mx.array) -> mx.array:
        return self.codebook.decode(codes)  # (B, T, dim)

    def encode(self, x: mx.array) -> mx.array:
        return self.codebook.encode(x)


class ResidualVectorQuantization(nn.Module):
    def __init__(self, num_quantizers: int, dim: int, codebook_size: int) -> None:
        super().__init__()
        self.layers = [VectorQuantization(dim, codebook_size) for _ in range(num_quantizers)]

    def decode(self, codes: mx.array) -> mx.array:
        """codes (B, Q, T) -> (B, T, dim)."""
        out = self.layers[0].decode(codes[:, 0])
        for q in range(1, codes.shape[1]):
            out = out + self.layers[q].decode(codes[:, q])
        return out

    def encode(self, x: mx.array, num_quantizers: int) -> mx.array:
        """x (B, T, dim) -> codes (B, num_quantizers, T)."""
        residual = x
        codes = []
        for layer in self.layers[:num_quantizers]:
            idx = layer.encode(residual)
            residual = residual - layer.decode(idx)
            codes.append(idx)
        return mx.stack(codes, axis=1)


class ResidualVectorQuantizer(nn.Module):
    """``rvq_first`` / ``rvq_rest``: 1x1 projections (Conv1d weights, kernel 1) around the codebooks."""

    def __init__(self, *, dimension: int, input_dimension: int, output_dimension: int, n_q: int, bins: int) -> None:
        super().__init__()
        self.input_proj = nn.Conv1d(input_dimension, dimension, 1, bias=False)
        self.output_proj = nn.Conv1d(dimension, output_dimension, 1, bias=False)
        self.vq = ResidualVectorQuantization(n_q, dimension, bins)

    def decode(self, codes: mx.array) -> mx.array:
        return self.output_proj(self.vq.decode(codes))  # (B, T, output_dim)

    def encode(self, x: mx.array, num_quantizers: int) -> mx.array:
        return self.vq.encode(self.input_proj(x), num_quantizers)


class SplitResidualVectorQuantizer(nn.Module):
    def __init__(self, *, dimension: int, input_dimension: int, output_dimension: int, n_q: int, n_q_semantic: int, bins: int) -> None:
        super().__init__()
        self.n_q_semantic = n_q_semantic
        self.rvq_first = ResidualVectorQuantizer(
            dimension=dimension, input_dimension=input_dimension, output_dimension=output_dimension, n_q=n_q_semantic, bins=bins
        )
        self.rvq_rest = ResidualVectorQuantizer(
            dimension=dimension, input_dimension=input_dimension, output_dimension=output_dimension, n_q=n_q - n_q_semantic, bins=bins
        )

    def decode(self, codes: mx.array) -> mx.array:
        """codes (B, Q, T) -> (B, T, output_dim)."""
        out = self.rvq_first.decode(codes[:, : self.n_q_semantic])
        if codes.shape[1] > self.n_q_semantic:
            out = out + self.rvq_rest.decode(codes[:, self.n_q_semantic :])
        return out


class Qwen3TTSCodecDecoder(nn.Module):
    """``decoder.*``: codes (B, Q, T) -> waveform (B, samples), decoded causally with persistent state."""

    def __init__(self, config: Qwen3TTSCodecDecoderConfig) -> None:
        super().__init__()
        self.config = config
        self.total_upsample = int(math.prod(list(config.upsample_rates) + list(config.upsampling_ratios)))
        self.pre_transformer = DecoderTransformer(config)
        self.quantizer = SplitResidualVectorQuantizer(
            dimension=config.codebook_dim // 2,
            input_dimension=config.codebook_dim,
            output_dimension=config.codebook_dim,
            n_q=config.num_quantizers,
            n_q_semantic=config.num_semantic_quantizers,
            bins=config.codebook_size,
        )
        self.pre_conv = CausalConv1d(config.codebook_dim, config.latent_dim, 3)
        self.upsample = [
            [CausalTransposeConv1d(config.latent_dim, config.latent_dim, factor, factor), ConvNeXtBlock(config.latent_dim)]
            for factor in config.upsampling_ratios
        ]
        output_dim = config.decoder_dim // (2 ** len(config.upsample_rates))
        self.decoder = [
            CausalConv1d(config.latent_dim, config.decoder_dim, 7),
            *[DecoderBlock(config, i) for i in range(len(config.upsample_rates))],
            SnakeBeta(output_dim),
            CausalConv1d(output_dim, 1, 7),
        ]

    def reset_state(self) -> None:
        self.pre_transformer.reset_state()
        self.pre_conv.reset_state()
        for up in self.upsample:
            up[0].reset_state()
            up[1].reset_state()
        self.decoder[0].reset_state()
        for block in self.decoder[1:-2]:
            block.reset_state()
        self.decoder[-1].reset_state()

    def step(self, codes: mx.array) -> mx.array:
        """Decode the NEXT ``codes`` (B, Q, T) of the current utterance -> (B, T * total_upsample) float32."""
        if codes.shape[1] != self.config.num_quantizers:
            raise ValueError(f"expected {self.config.num_quantizers} codebooks, got {codes.shape[1]}")
        x = self.quantizer.decode(codes)  # (B, T, codebook_dim)
        x = self.pre_conv.step(x)
        x = self.pre_transformer.step(x)
        for up in self.upsample:
            x = up[0].step(x)
            x = up[1].step(x)
        x = self.decoder[0].step(x)
        for block in self.decoder[1:-2]:
            x = block.step(x)
        x = self.decoder[-2](x)
        x = self.decoder[-1].step(x)
        return mx.clip(x[:, :, 0], -1.0, 1.0)

    def decode_all(self, codes: mx.array) -> mx.array:
        """Decode a whole utterance (B, Q, T) in one pass, from a fresh state."""
        self.reset_state()
        try:
            return self.step(codes)
        finally:
            self.reset_state()


# ============================================================================ encoder (HF Mimi layout)


class MimiConv1d(nn.Module):
    """``MimiConv1d``: causal conv with frame-aligning extra padding; NLC internally."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
        pad_mode: str = "constant",
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=0, dilation=dilation, bias=bias)
        self.stride = stride
        self.kernel_size = (kernel_size - 1) * dilation + 1
        self.padding_total = self.kernel_size - stride
        self.pad_mode = pad_mode

    def __call__(self, x: mx.array) -> mx.array:
        length = x.shape[1]
        n_frames = (length - self.kernel_size + self.padding_total) / self.stride + 1
        ideal = (math.ceil(n_frames) - 1) * self.stride + (self.kernel_size - self.padding_total)
        extra = max(0, ideal - length)
        left, right = self.padding_total, extra
        if left or right:
            if self.pad_mode == "constant":
                x = mx.pad(x, [(0, 0), (left, right), (0, 0)])
            elif self.pad_mode in ("replicate", "edge"):
                x = mx.pad(x, [(0, 0), (left, right), (0, 0)], mode="edge")
            else:
                raise ValueError(f"unsupported pad_mode {self.pad_mode!r}")
        return self.conv(x)


class _ELU(nn.Module):
    """Parameter-free placeholder keeping HF's ``layers.N`` numbering."""

    def __call__(self, x: mx.array) -> mx.array:
        return nn.elu(x, alpha=1.0)


class MimiResnetBlock(nn.Module):
    def __init__(self, config: Qwen3TTSCodecEncoderConfig, dim: int, dilations: List[int]) -> None:
        super().__init__()
        hidden = dim // config.compress
        kernel_sizes = (config.residual_kernel_size, 1)
        block = []
        for i, (k, d) in enumerate(zip(kernel_sizes, dilations)):
            in_ch = dim if i == 0 else hidden
            out_ch = dim if i == len(kernel_sizes) - 1 else hidden
            block.append(_ELU())
            block.append(MimiConv1d(in_ch, out_ch, k, dilation=d))
        self.block = block
        if config.use_conv_shortcut:
            raise NotImplementedError("Qwen3-TTS codec encoders use the identity shortcut")

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        for m in self.block:
            x = m(x)
        return residual + x


class MimiEncoder(nn.Module):
    """``encoder.encoder``: SEANet encoder as a flat layer list (HF numbering)."""

    def __init__(self, config: Qwen3TTSCodecEncoderConfig) -> None:
        super().__init__()
        layers: List[nn.Module] = [MimiConv1d(config.audio_channels, config.num_filters, config.kernel_size)]
        scaling = 1
        for ratio in reversed(config.upsampling_ratios):
            current = scaling * config.num_filters
            for j in range(config.num_residual_layers):
                layers.append(MimiResnetBlock(config, current, [config.dilation_growth_rate**j, 1]))
            layers.append(_ELU())
            layers.append(MimiConv1d(current, current * 2, 2 * ratio, stride=ratio))
            scaling *= 2
        layers.append(_ELU())
        layers.append(MimiConv1d(scaling * config.num_filters, config.hidden_size, config.last_kernel_size))
        self.layers = layers

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = layer(x)
        return x


class _MimiAttention(nn.Module):
    def __init__(self, config: Qwen3TTSCodecEncoderConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

    def __call__(self, x: mx.array, cos, sin, mask) -> mx.array:
        B, T, _ = x.shape
        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, T, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, T, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(B, T, -1))


class _MimiMLP(nn.Module):
    def __init__(self, config: Qwen3TTSCodecEncoderConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu(self.fc1(x)))


class _MimiTransformerLayer(nn.Module):
    def __init__(self, config: Qwen3TTSCodecEncoderConfig) -> None:
        super().__init__()
        self.self_attn = _MimiAttention(config)
        self.mlp = _MimiMLP(config)
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
        self.self_attn_layer_scale = _LayerScale(config.hidden_size, config.layer_scale_initial_scale)
        self.mlp_layer_scale = _LayerScale(config.hidden_size, config.layer_scale_initial_scale)

    def __call__(self, x: mx.array, cos, sin, mask) -> mx.array:
        x = x + self.self_attn_layer_scale(self.self_attn(self.input_layernorm(x), cos, sin, mask))
        return x + self.mlp_layer_scale(self.mlp(self.post_attention_layernorm(x)))


class MimiTransformer(nn.Module):
    """``encoder.encoder_transformer``: sliding-window causal transformer over (B, T, hidden)."""

    def __init__(self, config: Qwen3TTSCodecEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.layers = [_MimiTransformerLayer(config) for _ in range(config.num_hidden_layers)]

    def __call__(self, x: mx.array) -> mx.array:
        T = x.shape[1]
        cos, sin = _rope_cos_sin(mx.arange(T), self.config.head_dim, self.config.rope_theta, x.dtype)
        mask = _sliding_causal_mask(T, T, 0, self.config.sliding_window, x.dtype)
        for layer in self.layers:
            x = layer(x, cos, sin, mask)
        return x


class MimiSplitQuantizer(nn.Module):
    """``encoder.quantizer``: semantic (1) + acoustic (31) RVQs over the same input; encode only."""

    def __init__(self, config: Qwen3TTSCodecEncoderConfig) -> None:
        super().__init__()
        self.num_semantic_quantizers = config.num_semantic_quantizers
        self.semantic_residual_vector_quantizer = _MimiRVQ(config, config.num_semantic_quantizers)
        self.acoustic_residual_vector_quantizer = _MimiRVQ(config, config.num_quantizers - config.num_semantic_quantizers)

    def encode(self, x: mx.array, num_quantizers: int) -> mx.array:
        """x (B, T, hidden) -> codes (B, num_quantizers, T)."""
        codes = self.semantic_residual_vector_quantizer.encode(x, self.num_semantic_quantizers)
        if num_quantizers > self.num_semantic_quantizers:
            acoustic = self.acoustic_residual_vector_quantizer.encode(x, num_quantizers - self.num_semantic_quantizers)
            codes = mx.concatenate([codes, acoustic], axis=1)
        return codes


class _MimiRVQ(nn.Module):
    def __init__(self, config: Qwen3TTSCodecEncoderConfig, num_quantizers: int) -> None:
        super().__init__()
        self.input_proj = nn.Conv1d(config.hidden_size, config.codebook_dim, 1, bias=False)
        self.output_proj = nn.Conv1d(config.codebook_dim, config.hidden_size, 1, bias=False)
        self.layers = [_MimiVectorQuantization(config.codebook_dim, config.codebook_size) for _ in range(num_quantizers)]

    def encode(self, x: mx.array, num_quantizers: int) -> mx.array:
        residual = self.input_proj(x)
        codes = []
        for layer in self.layers[:num_quantizers]:
            idx = layer.codebook.encode(residual)
            residual = residual - layer.codebook.decode(idx)
            codes.append(idx)
        return mx.stack(codes, axis=1)


class _MimiVectorQuantization(nn.Module):
    def __init__(self, dim: int, codebook_size: int) -> None:
        super().__init__()
        self.codebook = _MimiCodebook(dim, codebook_size)


class _MimiCodebook(nn.Module):
    """HF ``MimiEuclideanCodebook``: parameters ``cluster_usage``, ``embed_sum``, ``initialized``."""

    def __init__(self, dim: int, codebook_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.initialized = mx.ones((1,))
        self.cluster_usage = mx.ones((codebook_size,))
        self.embed_sum = mx.zeros((codebook_size, dim))
        self._embedding: Optional[mx.array] = None

    def finalize(self) -> None:
        self._embedding = self.embed_sum / mx.maximum(self.cluster_usage, self.eps)[:, None]
        mx.eval(self._embedding)

    @property
    def embedding(self) -> mx.array:
        if self._embedding is None:
            self.finalize()
        return self._embedding

    def encode(self, x: mx.array) -> mx.array:
        shape = x.shape[:-1]
        flat = x.reshape(-1, x.shape[-1]).astype(mx.float32)
        table = self.embedding.astype(mx.float32)
        dists = mx.sum(flat * flat, axis=-1, keepdims=True) - 2.0 * (flat @ table.T) + mx.sum(table * table, axis=-1)[None, :]
        return mx.argmin(dists, axis=-1).reshape(shape)

    def decode(self, codes: mx.array) -> mx.array:
        return self.embedding[codes]


class Qwen3TTSCodecEncoder(nn.Module):
    """``encoder.*``: waveform (B, samples) at 24 kHz -> codes (B, 16, frames) at 12.5 Hz."""

    def __init__(self, config: Qwen3TTSCodecEncoderConfig, valid_num_quantizers: int) -> None:
        super().__init__()
        self.config = config
        self.valid_num_quantizers = int(valid_num_quantizers)
        self.encoder = MimiEncoder(config)
        self.encoder_transformer = MimiTransformer(config)
        encoder_frame_rate = config.sampling_rate / math.prod(config.upsampling_ratios)
        stride = int(encoder_frame_rate / config.frame_rate)
        self.downsample = MimiConv1d(config.hidden_size, config.hidden_size, 2 * stride, stride=stride, bias=False, pad_mode="replicate")
        self.quantizer = MimiSplitQuantizer(config)

    def encode(self, audio: mx.array) -> mx.array:
        x = audio[:, :, None].astype(mx.float32)  # (B, samples, 1)
        x = self.encoder(x)
        x = self.encoder_transformer(x)
        x = self.downsample(x)
        return self.quantizer.encode(x, self.valid_num_quantizers)


# ============================================================================ wrapper


class Qwen3TTSCodec(nn.Module):
    """The ``speech_tokenizer/`` checkpoint: ``decoder`` always, ``encoder`` when the config has one."""

    def __init__(self, config: Qwen3TTSCodecConfig, *, with_encoder: bool = True) -> None:
        super().__init__()
        self.config = config
        self.decoder = Qwen3TTSCodecDecoder(config.decoder_config)
        self.encoder = (
            Qwen3TTSCodecEncoder(config.encoder_config, config.encoder_valid_num_quantizers)
            if (with_encoder and config.encoder_config is not None)
            else None
        )

    @property
    def sample_rate(self) -> int:
        return int(self.config.output_sample_rate)

    @property
    def samples_per_frame(self) -> int:
        return int(self.config.decode_upsample_rate)

    def finalize(self) -> None:
        """Derive the codebook tables after the weights are loaded."""
        for layer in self.decoder.quantizer.rvq_first.vq.layers + self.decoder.quantizer.rvq_rest.vq.layers:
            layer.codebook.finalize()
        if self.encoder is not None:
            q = self.encoder.quantizer
            for layer in q.semantic_residual_vector_quantizer.layers + q.acoustic_residual_vector_quantizer.layers:
                layer.codebook.finalize()

    def encode(self, audio: mx.array) -> mx.array:
        """Mono 24 kHz waveform (samples,) -> codes (frames, 16)."""
        if self.encoder is None:
            raise RuntimeError("this Qwen3-TTS codec has no encoder (only Base checkpoints clone voices)")
        codes = self.encoder.encode(audio.reshape(1, -1))
        n_frames = -(-int(audio.shape[-1]) // int(self.config.encode_downsample_rate))
        return codes[0, :, :n_frames].T
