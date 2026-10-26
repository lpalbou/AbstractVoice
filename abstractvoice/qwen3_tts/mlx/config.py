# Derived from mlx-audio 0.6.1 (https://github.com/Blaizzy/mlx-audio, MIT, Copyright (c) 2025
# Prince Canuma and contributors), file mlx_audio/tts/models/qwen3_tts/config.py.
# Local modifications, listed exhaustively:
#   1. Configs are built from the ORIGINAL Qwen ``config.json`` / ``speech_tokenizer/config.json``
#      (``from_snapshot``); mlx-audio's ``BaseModelArgs``/``model_path`` plumbing is dropped.
#   2. ``Qwen3TTSTalkerConfig.spk_id`` is typed as ``Dict[str, int]`` (what the checkpoints hold).
#   3. The speech-tokenizer encoder config maps ``_frame_rate`` to ``frame_rate``.
#   4. Dataclasses are plain (no mlx import): this module is importable anywhere.
"""Dataclass views over the Qwen3-TTS snapshot configs, as the MLX port reads them."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Type, TypeVar

T = TypeVar("T")


def _only_fields(cls: Type[T], data: Dict[str, Any]) -> Dict[str, Any]:
    valid = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in valid}


@dataclass
class Qwen3TTSSpeakerEncoderConfig:
    """ECAPA-TDNN speaker encoder (Base checkpoints only)."""

    mel_dim: int = 128
    enc_dim: int = 1024
    enc_channels: List[int] = field(default_factory=lambda: [512, 512, 512, 512, 1536])
    enc_kernel_sizes: List[int] = field(default_factory=lambda: [5, 3, 3, 3, 1])
    enc_dilations: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 1])
    enc_attention_channels: int = 128
    enc_res2net_scale: int = 8
    enc_se_channels: int = 128
    sample_rate: int = 24000


@dataclass
class Qwen3TTSTalkerCodePredictorConfig:
    vocab_size: int = 2048
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 5
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 65536
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    attention_bias: bool = False
    layer_types: Optional[List[str]] = None
    num_code_groups: int = 16

    def __post_init__(self) -> None:
        if self.layer_types is None:
            self.layer_types = ["full_attention"] * int(self.num_hidden_layers)


@dataclass
class Qwen3TTSTalkerConfig:
    code_predictor_config: Optional[Qwen3TTSTalkerCodePredictorConfig] = None
    vocab_size: int = 3072
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    rope_scaling: Optional[Dict[str, Any]] = None
    attention_bias: bool = False
    num_code_groups: int = 16
    text_hidden_size: int = 2048
    text_vocab_size: int = 151936
    codec_eos_token_id: int = 2150
    codec_think_id: int = 2154
    codec_nothink_id: int = 2155
    codec_think_bos_id: int = 2156
    codec_think_eos_id: int = 2157
    codec_pad_id: int = 2148
    codec_bos_id: int = 2149
    codec_language_id: Optional[Dict[str, int]] = None
    spk_id: Optional[Dict[str, int]] = None
    spk_is_dialect: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.code_predictor_config is None:
            self.code_predictor_config = Qwen3TTSTalkerCodePredictorConfig()
        elif isinstance(self.code_predictor_config, dict):
            self.code_predictor_config = Qwen3TTSTalkerCodePredictorConfig(
                **_only_fields(Qwen3TTSTalkerCodePredictorConfig, self.code_predictor_config)
            )


@dataclass
class Qwen3TTSCodecDecoderConfig:
    attention_bias: bool = False
    latent_dim: int = 1024
    codebook_dim: int = 512
    codebook_size: int = 2048
    decoder_dim: int = 1536
    hidden_act: str = "silu"
    hidden_size: int = 512
    intermediate_size: int = 1024
    layer_scale_initial_scale: float = 0.01
    max_position_embeddings: int = 8000
    head_dim: int = 64
    num_attention_heads: int = 16
    num_hidden_layers: int = 8
    num_key_value_heads: int = 16
    num_quantizers: int = 16
    num_semantic_quantizers: int = 1
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    semantic_codebook_size: int = 4096
    sliding_window: int = 72
    upsample_rates: List[int] = field(default_factory=lambda: [8, 5, 4, 3])
    upsampling_ratios: List[int] = field(default_factory=lambda: [2, 2])
    vector_quantization_hidden_dimension: int = 512


@dataclass
class Qwen3TTSCodecEncoderConfig:
    frame_rate: float = 12.5
    attention_bias: bool = False
    audio_channels: int = 1
    codebook_dim: int = 256
    codebook_size: int = 2048
    compress: int = 2
    dilation_growth_rate: int = 2
    head_dim: int = 64
    hidden_act: str = "gelu"
    hidden_size: int = 512
    intermediate_size: int = 2048
    kernel_size: int = 7
    last_kernel_size: int = 3
    layer_scale_initial_scale: float = 0.01
    max_position_embeddings: int = 8000
    norm_eps: float = 1e-5
    num_attention_heads: int = 8
    num_filters: int = 64
    num_hidden_layers: int = 8
    num_key_value_heads: int = 8
    num_quantizers: int = 32
    num_residual_layers: int = 1
    num_semantic_quantizers: int = 1
    residual_kernel_size: int = 3
    rope_theta: float = 10000.0
    sampling_rate: int = 24000
    sliding_window: int = 250
    upsampling_ratios: List[int] = field(default_factory=lambda: [8, 6, 5, 4])
    use_causal_conv: bool = True
    use_conv_shortcut: bool = False
    vector_quantization_hidden_dimension: int = 256


@dataclass
class Qwen3TTSCodecConfig:
    encoder_config: Optional[Qwen3TTSCodecEncoderConfig] = None
    decoder_config: Optional[Qwen3TTSCodecDecoderConfig] = None
    encoder_valid_num_quantizers: int = 16
    input_sample_rate: int = 24000
    output_sample_rate: int = 24000
    decode_upsample_rate: int = 1920
    encode_downsample_rate: int = 1920

    def __post_init__(self) -> None:
        if isinstance(self.encoder_config, dict):
            data = dict(self.encoder_config)
            if "_frame_rate" in data and "frame_rate" not in data:
                data["frame_rate"] = data["_frame_rate"]
            self.encoder_config = Qwen3TTSCodecEncoderConfig(**_only_fields(Qwen3TTSCodecEncoderConfig, data))
        if self.decoder_config is None:
            self.decoder_config = Qwen3TTSCodecDecoderConfig()
        elif isinstance(self.decoder_config, dict):
            self.decoder_config = Qwen3TTSCodecDecoderConfig(
                **_only_fields(Qwen3TTSCodecDecoderConfig, self.decoder_config)
            )

    @classmethod
    def from_dir(cls, speech_tokenizer_dir: str) -> "Qwen3TTSCodecConfig":
        with open(os.path.join(speech_tokenizer_dir, "config.json"), "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return cls(**_only_fields(cls, data))


@dataclass
class Qwen3TTSModelConfig:
    """``config.json`` of a Qwen3-TTS snapshot, plus the codec config from ``speech_tokenizer/``."""

    talker_config: Optional[Qwen3TTSTalkerConfig] = None
    speaker_encoder_config: Optional[Qwen3TTSSpeakerEncoderConfig] = None
    tokenizer_type: str = "qwen3_tts_tokenizer_12hz"
    tts_model_size: str = "0b6"
    tts_model_type: str = "custom_voice"
    im_start_token_id: int = 151644
    im_end_token_id: int = 151645
    tts_pad_token_id: int = 151671
    tts_bos_token_id: int = 151672
    tts_eos_token_id: int = 151673
    codec: Optional[Qwen3TTSCodecConfig] = None

    def __post_init__(self) -> None:
        if self.talker_config is None:
            self.talker_config = Qwen3TTSTalkerConfig()
        elif isinstance(self.talker_config, dict):
            self.talker_config = Qwen3TTSTalkerConfig(**_only_fields(Qwen3TTSTalkerConfig, self.talker_config))
        if isinstance(self.speaker_encoder_config, dict):
            self.speaker_encoder_config = Qwen3TTSSpeakerEncoderConfig(
                **_only_fields(Qwen3TTSSpeakerEncoderConfig, self.speaker_encoder_config)
            )
        elif self.speaker_encoder_config is None and self.tts_model_type == "base":
            self.speaker_encoder_config = Qwen3TTSSpeakerEncoderConfig()

    @property
    def has_speaker_encoder(self) -> bool:
        return self.tts_model_type == "base"

    @property
    def sample_rate(self) -> int:
        return int(self.codec.output_sample_rate) if self.codec is not None else 24000

    @classmethod
    def from_snapshot(cls, snapshot_dir: str) -> "Qwen3TTSModelConfig":
        with open(os.path.join(snapshot_dir, "config.json"), "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if str(data.get("model_type") or "") != "qwen3_tts":
            raise ValueError(f"{snapshot_dir} is not a Qwen3-TTS checkpoint (model_type={data.get('model_type')!r})")
        cfg = cls(**_only_fields(cls, data))
        cfg.codec = Qwen3TTSCodecConfig.from_dir(os.path.join(snapshot_dir, "speech_tokenizer"))
        return cfg
