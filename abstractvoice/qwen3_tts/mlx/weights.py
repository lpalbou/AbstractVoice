"""Load the ORIGINAL Qwen3-TTS safetensors into the MLX modules, strictly, and quantize on load.

The PyTorch path loads with ``load_state_dict(strict=True)`` because a lenient loader once
assigned nothing and produced speech-shaped babble (see ``orchestration.load_strict_safetensors``).
The same discipline applies here: every checkpoint tensor must land on a module parameter of the
same shape and every parameter must be fed; anything else is a hard error, never a warning.

Key mapping is deliberately tiny -- the MLX modules are named after the PyTorch state dict -- and
limited to what MLX's layouts require:

* ``talker.`` / ``speaker_encoder.`` prefixes are dropped (one module each);
* ``Conv1d`` weights ``(out, in, k)`` become ``(out, k, in)``; ``ConvTranspose1d`` weights
  ``(in, out, k)`` become ``(out, k, in)``;
* the codec's ``._codebook.`` segment becomes ``.codebook.`` (MLX hides underscore attributes
  from ``parameters()``); the EMA statistics themselves are kept and the tables derived after load.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from ..backend import normalize_mlx_quantization

QUANT_GROUP_SIZE = 64

def _is_decoder_transposed_conv(name: str) -> bool:
    """The codec decoder's ConvTranspose1d weights: ``decoder.upsample.N.0.conv.weight`` and the
    upsampling conv of each vocoder block, ``decoder.decoder.N.block.1.conv.weight``."""
    parts = name.split(".")
    if parts[:2] == ["decoder", "upsample"] and parts[3:] == ["0", "conv", "weight"]:
        return True
    return parts[:2] == ["decoder", "decoder"] and parts[3:] == ["block", "1", "conv", "weight"]


def quantization_bits(quantization: str) -> Optional[int]:
    return {"bf16": None, "8bit": 8, "4bit": 4}[normalize_mlx_quantization(quantization)]


def load_safetensors(path: str) -> Dict[str, mx.array]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No model.safetensors at {path}; non-safetensors checkpoints are not supported.")
    return dict(mx.load(path))


def split_talker_checkpoint(raw: Dict[str, mx.array]) -> Tuple[Dict[str, mx.array], Dict[str, mx.array]]:
    """``model.safetensors`` -> (talker weights, speaker-encoder weights), both prefix-free."""
    talker: Dict[str, mx.array] = {}
    speaker: Dict[str, mx.array] = {}
    for key, value in raw.items():
        if key.startswith("talker."):
            talker[key[len("talker.") :]] = value
        elif key.startswith("speaker_encoder."):
            name = key[len("speaker_encoder.") :]
            if name.endswith(".weight") and value.ndim == 3:
                value = mx.transpose(value, (0, 2, 1))  # Conv1d (out, in, k) -> (out, k, in)
            speaker[name] = value.astype(mx.float32)
        else:
            raise ValueError(f"unexpected tensor {key!r} in the Qwen3-TTS checkpoint (neither talker nor speaker encoder)")
    return talker, speaker


def map_codec_checkpoint(raw: Dict[str, mx.array], *, with_encoder: bool) -> Dict[str, mx.array]:
    """``speech_tokenizer/model.safetensors`` -> MLX names and layouts."""
    out: Dict[str, mx.array] = {}
    for key, value in raw.items():
        if key.startswith("encoder."):
            if not with_encoder:
                continue
            if value.ndim == 3 and key.endswith(".weight"):
                value = mx.transpose(value, (0, 2, 1))
            out[key] = value.astype(mx.float32)
        elif key.startswith("decoder."):
            name = key.replace("._codebook.", ".codebook.")
            if value.ndim == 3 and name.endswith(".weight"):
                if _is_decoder_transposed_conv(name):
                    value = mx.transpose(value, (1, 2, 0))  # ConvTranspose1d (in, out, k) -> (out, k, in)
                else:
                    value = mx.transpose(value, (0, 2, 1))  # Conv1d (out, in, k) -> (out, k, in)
            out[name] = value.astype(mx.float32)
        else:
            raise ValueError(f"unexpected tensor {key!r} in the Qwen3-TTS speech_tokenizer checkpoint")
    return out


def load_weights_strict(module: nn.Module, weights: Dict[str, mx.array], *, label: str) -> None:
    """Assign ``weights`` to ``module`` with a full key/shape audit; raise on any drift."""
    expected = dict(tree_flatten(module.parameters()))
    missing = sorted(set(expected) - set(weights))
    unexpected = sorted(set(weights) - set(expected))
    if missing or unexpected:
        raise RuntimeError(
            f"Qwen3-TTS MLX {label}: checkpoint/model key drift. "
            f"missing={missing[:8]}{'...' if len(missing) > 8 else ''} "
            f"unexpected={unexpected[:8]}{'...' if len(unexpected) > 8 else ''}"
        )
    for key, param in expected.items():
        got = tuple(weights[key].shape)
        want = tuple(param.shape)
        if got != want:
            raise RuntimeError(f"Qwen3-TTS MLX {label}: shape mismatch for {key}: checkpoint {got}, model {want}")
    module.load_weights(list(weights.items()), strict=True)


def quantize_talker(talker: nn.Module, *, bits: Optional[int], group_size: int = QUANT_GROUP_SIZE) -> int:
    """Quantize the talker's Linear layers in place (embeddings stay bf16). Returns the layer count."""
    if bits is None:
        return 0
    count = {"n": 0}

    def predicate(path: str, module: nn.Module):
        if not isinstance(module, nn.Linear):
            return False
        if module.weight.shape[-1] % group_size != 0:
            return False
        count["n"] += 1
        return True

    nn.quantize(talker, group_size=group_size, bits=int(bits), class_predicate=predicate)
    return count["n"]


def parameter_bytes(module: nn.Module) -> int:
    return int(sum(v.nbytes for _, v in tree_flatten(module.parameters())))


def iter_arrays(modules: Iterable[nn.Module]):
    for module in modules:
        for _, value in tree_flatten(module.parameters()):
            yield value
