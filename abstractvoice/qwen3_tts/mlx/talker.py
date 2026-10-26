# Derived from mlx-audio 0.6.1 (https://github.com/Blaizzy/mlx-audio, MIT, Copyright (c) 2025
# Prince Canuma and contributors), file mlx_audio/tts/models/qwen3_tts/talker.py.
# Local modifications, listed exhaustively:
#   1. Rotary embeddings use ``mx.fast.rope``. The talker's interleaved MRoPE reduces to plain
#      RoPE here because the three position streams (temporal/height/width) are identical for a
#      text-and-codec sequence (see ``Qwen3TTSForConditionalGeneration.get_rope_index`` in the
#      vendored PyTorch model: "for pure text embedding sequence, the rotary position embedding
#      has no difference with modern LLMs"). This also avoids the fp32 matmul angle error of the
#      reference's ``inv_freq @ pos`` formulation at long positions.
#   2. The causal mask is ``mx.fast.scaled_dot_product_attention``'s ``"causal"`` mask; no
#      attention_mask / left-padding support (AbstractVoice synthesizes one utterance at a time).
#   3. ``KVCache`` comes from ``._cache``; ``sanitize`` is replaced by ``weights.py``.
#   4. Module attribute names mirror the PyTorch state dict (``talker.*`` keys with the
#      ``talker.`` prefix removed) so weights load by name, strictly -- with two fusions:
#      ``self_attn.{q,k,v}_proj`` -> ``self_attn.qkv_proj`` and ``mlp.{gate,up}_proj`` ->
#      ``mlp.gate_up_proj`` (weights concatenated at load). The per-frame decode loop is
#      kernel-launch-bound; two matmuls per layer fewer is ~15-20% per frame.
"""Talker (first codebook) and code predictor (residual codebooks) of Qwen3-TTS."""

from __future__ import annotations

from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from ._cache import KVCache
from .config import Qwen3TTSTalkerCodePredictorConfig, Qwen3TTSTalkerConfig


class _Attention(nn.Module):
    """GQA attention with QK-norm and RoPE (Qwen3 style)."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rms_norm_eps: float,
        rope_theta: float,
        bias: bool,
    ) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.scale = self.head_dim**-0.5
        self.rope_theta = float(rope_theta)
        # One matmul for q, k and v (the checkpoint's three projections are concatenated at
        # load, see weights.py): the decode loop is launch-bound, not FLOP-bound.
        q_out = self.num_heads * self.head_dim
        kv_out = self.num_kv_heads * self.head_dim
        self._splits = [q_out, q_out + kv_out]
        self.qkv_proj = nn.Linear(hidden_size, q_out + 2 * kv_out, bias=bias)
        self.o_proj = nn.Linear(q_out, hidden_size, bias=bias)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=rms_norm_eps)

    def __call__(self, x: mx.array, cache: Optional[KVCache], mask) -> mx.array:
        B, T, _ = x.shape
        q, k, v = mx.split(self.qkv_proj(x), self._splits, axis=-1)
        q = self.q_norm(q.reshape(B, T, self.num_heads, self.head_dim)).transpose(0, 2, 1, 3)
        k = self.k_norm(k.reshape(B, T, self.num_kv_heads, self.head_dim)).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        offset = cache.offset if cache is not None else 0
        q = mx.fast.rope(q, self.head_dim, traditional=False, base=self.rope_theta, scale=1.0, offset=offset)
        k = mx.fast.rope(k, self.head_dim, traditional=False, base=self.rope_theta, scale=1.0, offset=offset)
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(B, T, -1))


class _MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        # gate and up in one matmul (concatenated at load, see weights.py).
        self.gate_up_proj = nn.Linear(hidden_size, 2 * intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        gate, up = mx.split(self.gate_up_proj(x), 2, axis=-1)
        return self.down_proj(nn.silu(gate) * up)


class _DecoderLayer(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rms_norm_eps: float,
        rope_theta: float,
        attention_bias: bool,
    ) -> None:
        super().__init__()
        self.self_attn = _Attention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rms_norm_eps=rms_norm_eps,
            rope_theta=rope_theta,
            bias=attention_bias,
        )
        self.mlp = _MLP(hidden_size, intermediate_size)
        self.input_layernorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)

    def __call__(self, x: mx.array, cache: Optional[KVCache], mask) -> mx.array:
        x = x + self.self_attn(self.input_layernorm(x), cache, mask)
        return x + self.mlp(self.post_attention_layernorm(x))


def _causal_mask(seq_len: int):
    return "causal" if seq_len > 1 else None


class ResizeMLP(nn.Module):
    """Text-embedding projection into the talker width (``talker.text_projection``)."""

    def __init__(self, input_size: int, intermediate_size: int, output_size: int) -> None:
        super().__init__()
        self.linear_fc1 = nn.Linear(input_size, intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(intermediate_size, output_size, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear_fc2(nn.silu(self.linear_fc1(x)))


class Qwen3TTSTalkerModel(nn.Module):
    """Talker transformer body (``talker.model.*``)."""

    def __init__(self, config: Qwen3TTSTalkerConfig) -> None:
        super().__init__()
        self.config = config
        self.codec_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.text_embedding = nn.Embedding(config.text_vocab_size, config.text_hidden_size)
        self.layers = [
            _DecoderLayer(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                rms_norm_eps=config.rms_norm_eps,
                rope_theta=config.rope_theta,
                attention_bias=config.attention_bias,
            )
            for _ in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, inputs_embeds: mx.array, cache: Optional[List[KVCache]]) -> mx.array:
        mask = _causal_mask(inputs_embeds.shape[1])
        x = inputs_embeds
        for i, layer in enumerate(self.layers):
            x = layer(x, cache[i] if cache is not None else None, mask)
        return self.norm(x)

    def make_cache(self) -> List[KVCache]:
        return [KVCache() for _ in self.layers]


class CodePredictorModel(nn.Module):
    """Code predictor body (``talker.code_predictor.model.*``)."""

    def __init__(self, config: Qwen3TTSTalkerCodePredictorConfig, talker_hidden_size: int) -> None:
        super().__init__()
        self.config = config
        # One embedding table per residual codebook (indexed 0 .. num_code_groups-2).
        self.codec_embedding = [
            nn.Embedding(config.vocab_size, talker_hidden_size) for _ in range(config.num_code_groups - 1)
        ]
        self.layers = [
            _DecoderLayer(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                rms_norm_eps=config.rms_norm_eps,
                rope_theta=config.rope_theta,
                attention_bias=config.attention_bias,
            )
            for _ in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, inputs_embeds: mx.array, cache: Optional[List[KVCache]]) -> mx.array:
        mask = _causal_mask(inputs_embeds.shape[1])
        x = inputs_embeds
        for i, layer in enumerate(self.layers):
            x = layer(x, cache[i] if cache is not None else None, mask)
        return self.norm(x)

    def make_cache(self) -> List[KVCache]:
        return [KVCache() for _ in self.layers]


class Qwen3TTSTalkerCodePredictor(nn.Module):
    """Residual-codebook predictor (``talker.code_predictor.*``)."""

    def __init__(self, config: Qwen3TTSTalkerCodePredictorConfig, talker_hidden_size: int) -> None:
        super().__init__()
        self.config = config
        self.num_code_groups = int(config.num_code_groups)
        # 1.7B talkers are 2048 wide while the predictor stays 1024 wide.
        if int(config.hidden_size) != int(talker_hidden_size):
            self.small_to_mtp_projection = nn.Linear(talker_hidden_size, config.hidden_size, bias=True)
        else:
            self.small_to_mtp_projection = None
        self.model = CodePredictorModel(config, talker_hidden_size)
        self.lm_head = [
            nn.Linear(config.hidden_size, config.vocab_size, bias=False) for _ in range(config.num_code_groups - 1)
        ]

    @property
    def codec_embedding(self) -> List[nn.Embedding]:
        return self.model.codec_embedding

    def __call__(self, inputs_embeds: mx.array, cache: Optional[List[KVCache]], generation_step: int) -> mx.array:
        if self.small_to_mtp_projection is not None:
            inputs_embeds = self.small_to_mtp_projection(inputs_embeds)
        x = self.model(inputs_embeds, cache)
        return self.lm_head[generation_step](x)

    def make_cache(self) -> List[KVCache]:
        return self.model.make_cache()


class Qwen3TTSTalkerForConditionalGeneration(nn.Module):
    """``talker.*``: text projection, talker body, first-codebook head and code predictor."""

    def __init__(self, config: Qwen3TTSTalkerConfig) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen3TTSTalkerModel(config)
        self.text_projection = ResizeMLP(config.text_hidden_size, config.text_hidden_size, config.hidden_size)
        self.codec_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.code_predictor = Qwen3TTSTalkerCodePredictor(config.code_predictor_config, config.hidden_size)

    def embed_codec(self, ids: mx.array) -> mx.array:
        return self.model.codec_embedding(ids)

    def embed_text(self, ids: mx.array) -> mx.array:
        return self.text_projection(self.model.text_embedding(ids))

    def __call__(self, inputs_embeds: mx.array, cache: Optional[List[KVCache]]) -> Tuple[mx.array, mx.array]:
        hidden = self.model(inputs_embeds, cache)
        return self.codec_head(hidden), hidden

    def make_cache(self) -> List[KVCache]:
        return self.model.make_cache()
