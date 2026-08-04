# Derived from qwen-tts 0.1.1 (https://github.com/QwenLM/Qwen3-TTS, Apache-2.0),
# upstream file: qwen_tts/inference/qwen3_tts_tokenizer.py
# Local modifications, listed exhaustively; encode/decode bodies are upstream-verbatim:
#   1. 12Hz family only: the 25Hz (v1) classes, their sox/whisper deps, and the
#      25Hz decode branch are dropped.
#   2. No Auto* registration: config/model/feature extractor are constructed from
#      their concrete classes, so loading has no global registry side effects.
#   3. URL/base64/librosa audio loading removed. Audio in is a local file path
#      (read via soundfile) or a (np.ndarray, sr) pair; resampling uses
#      abstractvoice.audio.resample.sinc_resample_mono, which band-limits --
#      np.interp-style resampling folds >Nyquist energy into the codec input.
#   4. Class renamed Qwen3TTSTokenizer -> Qwen3TTSCodec (it is a neural audio
#      codec; "tokenizer" collides with the text tokenizer in this package).
"""Speech codec wrapper for the Qwen3-TTS 12Hz family.

Loads the ``speech_tokenizer/`` subfolder that every Qwen3-TTS model repo
bundles, and exposes the two operations the model needs: ``encode`` (waveform ->
discrete codes, for voice cloning prompts) and ``decode`` (codes -> waveform).
"""

from __future__ import annotations

import json
import os
from typing import Any, List, Optional, Tuple, Union

import numpy as np

AudioInput = Union[str, np.ndarray, Tuple[np.ndarray, int], list]


class Qwen3TTSCodec:
    """12Hz speech codec with HuggingFace-style loading, dependency-light."""

    def __init__(self):
        self.model = None
        self.feature_extractor = None
        self.config = None
        self.device = None

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs) -> "Qwen3TTSCodec":
        """Load the codec from a local ``speech_tokenizer/`` directory.

        ``kwargs`` (e.g. a talker-level ``dtype``) are accepted for signature
        compatibility but deliberately ignored: the codec ships fp32 weights and
        its residual-quantizer decode divides by an epsilon of 1e-5, which
        overflows in fp16 -- it stays fp32 unless you have measured otherwise.
        """
        import torch
        from transformers import EncodecFeatureExtractor

        from .configuration_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Config
        from .modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Model

        inst = cls()

        path = str(pretrained_model_name_or_path)
        preprocessor_path = os.path.join(path, "preprocessor_config.json")
        with open(preprocessor_path, "r", encoding="utf-8") as fh:
            preprocessor_cfg = json.load(fh)
        preprocessor_cfg.pop("feature_extractor_type", None)
        inst.feature_extractor = EncodecFeatureExtractor(**preprocessor_cfg)

        inst.config = Qwen3TTSTokenizerV2Config.from_pretrained(path)
        # The codec checkpoint is fp32 and its RVQ decode divides by 1e-5. A
        # caller-supplied bf16/fp16 dtype would ROUND THE WEIGHTS AT LOAD -- an
        # upcast afterwards cannot recover them -- so the codec pins fp32,
        # whatever dtype the talker load passed down. Weights load through the
        # same explicit strict path as the main model (load_strict_safetensors):
        # transformers 5.8's stock loader silently assigned nothing for the
        # composite class, and "worked for the codec today" is not a guarantee.
        _ = kwargs
        from .orchestration import load_strict_safetensors

        inst.model = Qwen3TTSTokenizerV2Model(inst.config)
        load_strict_safetensors(inst.model, path)
        inst.model = inst.model.to(dtype=torch.float32)
        inst.model.eval()

        inst.device = getattr(inst.model, "device", None)
        if inst.device is None:
            try:
                inst.device = next(inst.model.parameters()).device
            except StopIteration:
                inst.device = torch.device("cpu")

        return inst

    def to(self, device: Any) -> "Qwen3TTSCodec":
        self.model = self.model.to(device)
        self.device = device
        return self

    # ------------------------------------------------------------- audio input

    def load_audio(self, x: str, target_sr: int) -> np.ndarray:
        """Load a local audio file as mono float32 at ``target_sr``."""
        import soundfile as sf

        from ..audio.resample import sinc_resample_mono

        audio, sr = sf.read(x, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)
        if int(sr) != int(target_sr):
            audio = sinc_resample_mono(audio, int(sr), int(target_sr))
        return audio.astype(np.float32)

    def _normalize_audio_inputs(self, audios: AudioInput, sr: Optional[int]) -> List[np.ndarray]:
        """Normalize to a list of mono float32 waveforms at the codec input rate."""
        from ..audio.resample import sinc_resample_mono

        target_sr = int(self.feature_extractor.sampling_rate)

        if isinstance(audios, (str, np.ndarray)) or (
            isinstance(audios, tuple) and len(audios) == 2 and isinstance(audios[0], np.ndarray)
        ):
            audios = [audios]

        if len(audios) == 0:
            return []

        out: List[np.ndarray] = []
        for a in audios:
            if isinstance(a, str):
                out.append(self.load_audio(a, target_sr=target_sr))
                continue
            if isinstance(a, tuple) and len(a) == 2 and isinstance(a[0], np.ndarray):
                wav, wav_sr = a[0], int(a[1])
            elif isinstance(a, np.ndarray):
                if sr is None:
                    raise ValueError("For numpy waveform input, you must provide `sr` (original sampling rate).")
                wav, wav_sr = a, int(sr)
            else:
                raise TypeError(f"Unsupported audio input type: {type(a)}")
            if wav.ndim > 1:
                wav = np.mean(wav, axis=-1)
            if wav_sr != target_sr:
                wav = sinc_resample_mono(wav.astype(np.float32), wav_sr, target_sr)
            out.append(wav.astype(np.float32))
        return out

    # ------------------------------------------------------------ encode/decode

    def encode(self, audios: AudioInput, sr: Optional[int] = None, return_dict: bool = True):
        """Batch-encode audio into discrete codes (12Hz: (codes_len, num_quantizers))."""
        import torch

        wavs = self._normalize_audio_inputs(audios, sr=sr)

        inputs = self.feature_extractor(
            raw_audio=wavs,
            sampling_rate=int(self.feature_extractor.sampling_rate),
            return_tensors="pt",
        )
        inputs = inputs.to(self.device).to(self.model.dtype)

        with torch.inference_mode():
            # model.encode expects (B, T) and (B, T)
            enc = self.model.encode(
                inputs["input_values"].squeeze(1),
                inputs["padding_mask"].squeeze(1),
                return_dict=return_dict,
            )
        return enc

    def decode(self, encoded) -> Tuple[List[np.ndarray], int]:
        """Decode codes back to waveforms; returns (list of float32 arrays, sample rate)."""
        import torch
        from torch.nn.utils.rnn import pad_sequence

        def _to_tensor(x, dtype=None):
            if isinstance(x, torch.Tensor):
                return x
            x = np.asarray(x)
            t = torch.from_numpy(x)
            if dtype is not None:
                t = t.to(dtype)
            return t

        if hasattr(encoded, "audio_codes"):
            audio_codes_list = encoded.audio_codes
        elif isinstance(encoded, dict):
            audio_codes_list = encoded["audio_codes"]
        elif isinstance(encoded, list):
            audio_codes_list = [e["audio_codes"] for e in encoded]
        else:
            raise TypeError("`encoded` must be an encode output, a dict, or a list of dicts.")

        if isinstance(audio_codes_list, torch.Tensor):
            t = audio_codes_list
            if t.dim() == 2:
                # 12Hz single sample: (C, Q) -> (1, C, Q)
                t = t.unsqueeze(0)
            audio_codes_padded = t.to(self.device)
        else:
            audio_codes_list = [_to_tensor(c, dtype=torch.long) for c in audio_codes_list]
            audio_codes_padded = pad_sequence(audio_codes_list, batch_first=True, padding_value=-1).to(self.device)

        with torch.inference_mode():
            dec = self.model.decode(audio_codes_padded, return_dict=True)
            wav_tensors = dec.audio_values

        wavs = [w.to(torch.float32).detach().cpu().numpy() for w in wav_tensors]
        return wavs, int(self.model.get_output_sample_rate())

    # --------------------------------------------------------------- metadata

    def get_model_type(self) -> str:
        return self.model.get_model_type()

    def get_input_sample_rate(self) -> int:
        return int(self.model.get_input_sample_rate())

    def get_output_sample_rate(self) -> int:
        return int(self.model.get_output_sample_rate())

    def get_encode_downsample_rate(self) -> int:
        return int(self.model.get_encode_downsample_rate())

    def get_decode_upsample_rate(self) -> int:
        return int(self.model.get_decode_upsample_rate())
