"""Qwen3-TTS on Apple silicon through MLX (ADR 0008).

An MLX port of the 12Hz Qwen3-TTS family (talker + code predictor, speech
codec, speaker encoder) that loads the ORIGINAL Qwen bf16 safetensors from the
same Hugging Face snapshot the PyTorch path uses and quantizes on load. The
model code is derived from mlx-audio's implementation (MIT; see
``third_party_licenses/mlx_audio_notice.txt`` and each file's header for the
local modifications) and from the Kyutai Mimi modules it builds the codec
encoder on.

Public entry point: :class:`abstractvoice.qwen3_tts.mlx.model.Qwen3TTSMLXModel`,
driven by :class:`abstractvoice.qwen3_tts.runtime.Qwen3TTSRuntime` when the
backend policy (``abstractvoice.qwen3_tts.backend``) selects ``mlx``.

Importing this package must stay cheap: nothing here imports ``mlx``. Import
the submodules explicitly (``.model``, ``.tokenizer``, ...).
"""

from __future__ import annotations

__all__ = []
