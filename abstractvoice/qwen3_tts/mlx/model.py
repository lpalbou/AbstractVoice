# Prompt layouts and the generation loop are derived from mlx-audio 0.6.1
# (https://github.com/Blaizzy/mlx-audio, MIT, Copyright (c) 2025 Prince Canuma and contributors),
# file mlx_audio/tts/models/qwen3_tts/qwen3_tts.py, re-aligned on the vendored PyTorch model
# (``modeling_qwen3_tts.Qwen3TTSForConditionalGeneration.generate``), which is the contract the
# checkpoints were trained against. Local modifications, listed exhaustively:
#   1. Prompt layouts follow upstream's generate() for BOTH text-feeding modes: ``non_streaming_mode``
#      (all text prefilled; CustomVoice/VoiceDesign default) and streaming mode (first text token in
#      the prefill, the rest fed one embedding per frame; the voice-clone default). mlx-audio only
#      implemented the former and used a different ICL layout.
#   2. Sampling mirrors the Hugging Face generation stack the PyTorch path runs: the repetition
#      penalty covers the WHOLE generated history (mlx-audio: a 64-token window), EOS is suppressed
#      for the first two frames (``min_new_tokens=2``), the code predictor uses the separate
#      ``subtalker_*`` settings, ``do_sample=False`` is greedy; warpers run temperature -> top-k -> top-p.
#   3. Frame-level streaming yields CONTIGUOUS waveform chunks from one stateful codec decode (no
#      overlap, no context re-decoding, no fades); the ICL reference codes prime the decoder state and
#      their audio is dropped, which is upstream's proportional cut computed exactly.
#   4. Single-utterance only (no batching, no progress bars, no result dataclasses); audio comes back
#      as NumPy float32 so the caller never touches ``mlx`` types.
"""The Qwen3-TTS model on MLX: loading, prompting, generation, streaming."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

import mlx.core as mx
import numpy as np

from .codec import Qwen3TTSCodec
from .config import Qwen3TTSModelConfig
from .speaker_encoder import Qwen3TTSSpeakerEncoder, mel_spectrogram
from .talker import Qwen3TTSTalkerForConditionalGeneration
from .tokenizer import Qwen3TTSTextTokenizer
from ..backend import resolve_mlx_quantization
from .weights import (
    load_safetensors,
    load_weights_strict,
    map_codec_checkpoint,
    parameter_bytes,
    quantization_bits,
    quantize_talker,
    split_talker_checkpoint,
)

FRAMES_PER_SECOND = 12.5
# Upstream generate(): EOS cannot be sampled before two frames exist (min_new_tokens=2).
MIN_NEW_FRAMES = 2


@dataclass
class SamplingParams:
    """The talker/predictor sampling knobs, named as in ``Qwen3TTSSettings.generate_kwargs()``."""

    do_sample: bool = True
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 1.0
    repetition_penalty: float = 1.05
    subtalker_dosample: bool = True
    subtalker_temperature: float = 0.9
    subtalker_top_k: int = 50
    subtalker_top_p: float = 1.0

    @classmethod
    def from_generate_kwargs(cls, kwargs: Dict[str, Any]) -> "SamplingParams":
        names = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in kwargs.items() if k in names})


@dataclass
class GenerationPrompt:
    """Talker inputs for one utterance (see ``Qwen3TTSMLXModel.*_prompt``)."""

    input_embeds: mx.array  # (1, L, hidden)
    trailing_text_hidden: mx.array  # (1, T, hidden): text embeddings fed one per frame, then tts_pad
    tts_pad_embed: mx.array  # (1, 1, hidden)
    ref_codes: Optional[mx.array] = None  # (T_ref, 16) ICL prefix the codec decoder must see first
    text: str = ""


@dataclass
class MLXClonePrompt:
    """A voice-clone prompt built by a Base checkpoint (the MLX counterpart of ``VoiceClonePromptItem``)."""

    ref_codes: Optional[mx.array]  # (T, 16) int32, None in x-vector-only mode
    speaker_embedding: mx.array  # (enc_dim,) float32
    x_vector_only_mode: bool
    icl_mode: bool
    ref_text: Optional[str] = None


@dataclass
class LoadReport:
    snapshot_dir: str
    quantization: str
    seconds: Dict[str, float] = field(default_factory=dict)
    quantized_linear_layers: int = 0
    talker_bytes: int = 0
    codec_bytes: int = 0
    speaker_encoder_bytes: int = 0
    prime_seconds: Optional[float] = None


def _sample(logits: mx.array, *, do_sample: bool, temperature: float, top_k: int, top_p: float) -> mx.array:
    """One token from ``logits`` (1, V) float32 -> (1, 1) int32, Hugging Face warper semantics."""
    if not do_sample:
        return mx.argmax(logits, axis=-1, keepdims=True).astype(mx.int32)
    if temperature != 1.0:
        logits = logits / float(temperature)
    vocab = logits.shape[-1]
    if 0 < int(top_k) < vocab:
        kth = mx.min(mx.topk(logits, int(top_k), axis=-1), axis=-1, keepdims=True)
        logits = mx.where(logits < kth, mx.array(-mx.inf, logits.dtype), logits)
    if 0.0 < float(top_p) < 1.0:
        probs = mx.softmax(logits, axis=-1)
        order = mx.argsort(-probs, axis=-1)
        sorted_probs = mx.take_along_axis(probs, order, axis=-1)
        cumulative = mx.cumsum(sorted_probs, axis=-1)
        keep_sorted = (cumulative - sorted_probs) < float(top_p)
        keep = mx.put_along_axis(mx.zeros_like(keep_sorted), order, keep_sorted, axis=-1)
        logits = mx.where(keep, logits, mx.array(-mx.inf, logits.dtype))
    return mx.random.categorical(logits, axis=-1)[:, None].astype(mx.int32)


class Qwen3TTSMLXModel:
    """One Qwen3-TTS snapshot on MLX: CustomVoice, VoiceDesign and Base (cloning) checkpoints."""

    def __init__(
        self,
        *,
        config: Qwen3TTSModelConfig,
        talker: Qwen3TTSTalkerForConditionalGeneration,
        codec: Qwen3TTSCodec,
        speaker_encoder: Optional[Qwen3TTSSpeakerEncoder],
        tokenizer: Qwen3TTSTextTokenizer,
        report: LoadReport,
    ) -> None:
        self.config = config
        self.talker = talker
        self.codec = codec
        self.speaker_encoder = speaker_encoder
        self.tokenizer = tokenizer
        self.report = report
        talker_cfg = config.talker_config
        vocab = int(talker_cfg.vocab_size)
        eos = int(talker_cfg.codec_eos_token_id)
        bias = np.zeros((vocab,), dtype=np.float32)
        bias[vocab - 1024 : vocab] = -np.inf  # special codec ids are never sampled...
        bias[eos] = 0.0  # ...except EOS
        self._suppress_bias = mx.array(bias)
        eos_bias = np.zeros((vocab,), dtype=np.float32)
        eos_bias[eos] = -np.inf
        self._eos_bias = mx.array(eos_bias)
        self._vocab_ids = mx.arange(vocab)

    # ------------------------------------------------------------------ loading

    @classmethod
    def from_snapshot(cls, snapshot_dir: str, *, quantization: str = "8bit") -> "Qwen3TTSMLXModel":
        quantization = resolve_mlx_quantization(quantization)
        bits = quantization_bits(quantization)
        report = LoadReport(snapshot_dir=str(snapshot_dir), quantization=quantization)
        t_start = time.perf_counter()

        config = Qwen3TTSModelConfig.from_snapshot(snapshot_dir)
        tokenizer = Qwen3TTSTextTokenizer(snapshot_dir)
        report.seconds["config_tokenizer"] = time.perf_counter() - t_start

        t0 = time.perf_counter()
        raw = load_safetensors(os.path.join(snapshot_dir, "model.safetensors"))
        talker_weights, speaker_weights = split_talker_checkpoint(raw)
        talker = Qwen3TTSTalkerForConditionalGeneration(config.talker_config)
        load_weights_strict(talker, talker_weights, label="talker")
        report.quantized_linear_layers = quantize_talker(talker, bits=bits)
        speaker_encoder = None
        if config.has_speaker_encoder:
            speaker_encoder = Qwen3TTSSpeakerEncoder(config.speaker_encoder_config)
            load_weights_strict(speaker_encoder, speaker_weights, label="speaker encoder")
        elif speaker_weights:
            raise RuntimeError("checkpoint carries speaker_encoder weights but config.tts_model_type is not 'base'")
        mx.eval(talker.parameters())
        if speaker_encoder is not None:
            mx.eval(speaker_encoder.parameters())
        report.seconds["talker"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        codec = Qwen3TTSCodec(config.codec, with_encoder=config.has_speaker_encoder)
        codec_raw = load_safetensors(os.path.join(snapshot_dir, "speech_tokenizer", "model.safetensors"))
        load_weights_strict(codec, map_codec_checkpoint(codec_raw, with_encoder=codec.encoder is not None), label="codec")
        mx.eval(codec.parameters())
        codec.finalize()
        report.seconds["codec"] = time.perf_counter() - t0

        report.talker_bytes = parameter_bytes(talker)
        report.codec_bytes = parameter_bytes(codec)
        report.speaker_encoder_bytes = parameter_bytes(speaker_encoder) if speaker_encoder is not None else 0
        report.seconds["total"] = time.perf_counter() - t_start
        return cls(config=config, talker=talker, codec=codec, speaker_encoder=speaker_encoder, tokenizer=tokenizer, report=report)

    # ------------------------------------------------------------------ metadata

    @property
    def model_type(self) -> str:
        return str(self.config.tts_model_type)

    @property
    def model_size(self) -> str:
        return str(self.config.tts_model_size)

    @property
    def sample_rate(self) -> int:
        return int(self.config.sample_rate)

    @property
    def samples_per_frame(self) -> int:
        return int(self.codec.samples_per_frame)

    def supported_speakers(self) -> List[str]:
        return sorted(str(s).lower() for s in (self.config.talker_config.spk_id or {}))

    def supported_languages(self) -> List[str]:
        langs = ["auto"] + [str(k).lower() for k in (self.config.talker_config.codec_language_id or {}) if "dialect" not in str(k)]
        return sorted(langs)

    # ------------------------------------------------------------------ prompts

    def _text_ids(self, text: str) -> mx.array:
        return mx.array(self.tokenizer.encode(text), dtype=mx.int32)[None, :]

    def _embed_text(self, ids: mx.array) -> mx.array:
        return self.talker.embed_text(ids)

    def _embed_codec(self, ids) -> mx.array:
        return self.talker.embed_codec(mx.array(ids, dtype=mx.int32) if not isinstance(ids, mx.array) else ids)

    def _language_id(self, language: Optional[str], speaker: Optional[str]) -> Optional[int]:
        cfg = self.config.talker_config
        lang = str(language or "auto").strip().lower()
        if lang not in self.supported_languages():
            raise ValueError(f"Unsupported languages: [{language!r}]. Supported: {self.supported_languages()}")
        language_id = None if lang == "auto" else int(cfg.codec_language_id[lang])
        if lang in ("chinese", "auto") and speaker:
            dialect = (cfg.spk_is_dialect or {}).get(str(speaker).lower())
            if dialect:
                language_id = int(cfg.codec_language_id[str(dialect)])
        return language_id

    def _codec_prefix(self, language_id: Optional[int], speaker_embed: Optional[mx.array]) -> mx.array:
        """``codec_input_embedding``: think/nothink block, optional speaker vector, codec pad + bos."""
        cfg = self.config.talker_config
        if language_id is None:
            prefill = [cfg.codec_nothink_id, cfg.codec_think_bos_id, cfg.codec_think_eos_id]
        else:
            prefill = [cfg.codec_think_id, cfg.codec_think_bos_id, int(language_id), cfg.codec_think_eos_id]
        head = self._embed_codec([prefill])
        tail = self._embed_codec([[cfg.codec_pad_id, cfg.codec_bos_id]])
        if speaker_embed is None:
            return mx.concatenate([head, tail], axis=1)
        return mx.concatenate([head, speaker_embed.reshape(1, 1, -1).astype(head.dtype), tail], axis=1)

    def _assistant_ids(self, text: str) -> mx.array:
        return self._text_ids(f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n")

    def _build_prompt(
        self,
        *,
        text: str,
        language: Optional[str],
        speaker: Optional[str],
        speaker_embed: Optional[mx.array],
        instruct: Optional[str],
        non_streaming_mode: bool,
        icl: Optional[MLXClonePrompt] = None,
    ) -> GenerationPrompt:
        cfg = self.config.talker_config
        input_id = self._assistant_ids(str(text))
        tts_embeds = self._embed_text(
            mx.array([[self.config.tts_bos_token_id, self.config.tts_eos_token_id, self.config.tts_pad_token_id]], dtype=mx.int32)
        )
        tts_bos, tts_eos, tts_pad = tts_embeds[:, 0:1, :], tts_embeds[:, 1:2, :], tts_embeds[:, 2:3, :]

        language_id = self._language_id(language, speaker)
        codec_input = self._codec_prefix(language_id, speaker_embed)

        role = self._embed_text(input_id[:, :3])
        pad_count = codec_input.shape[1] - 2
        prefix = mx.concatenate([mx.broadcast_to(tts_pad, (1, pad_count, tts_pad.shape[-1])), tts_bos], axis=1) + codec_input[:, :-1, :]
        parts: List[mx.array] = []
        if instruct:
            parts.append(self._embed_text(self._text_ids(f"<|im_start|>user\n{instruct}<|im_end|>\n")))
        parts.append(role)
        parts.append(prefix)

        ref_codes = None
        if icl is not None and icl.ref_codes is not None and icl.icl_mode:
            ref_id = self._text_ids(f"<|im_start|>assistant\n{icl.ref_text}<|im_end|>\n")[:, 3:-2]
            icl_embed, trailing = self._icl_prompt(
                text_id=input_id[:, 3:-5],
                ref_id=ref_id,
                ref_codes=icl.ref_codes,
                tts_pad=tts_pad,
                tts_eos=tts_eos,
                non_streaming_mode=non_streaming_mode,
            )
            parts.append(icl_embed)
            ref_codes = icl.ref_codes
        else:
            text_ids = input_id[:, 3:-5]
            if non_streaming_mode:
                text_embed = mx.concatenate([self._embed_text(text_ids), tts_eos], axis=1)
                codec_pad = self._embed_codec([[cfg.codec_pad_id]])
                parts.append(text_embed + codec_pad)
                parts.append(tts_pad + self._embed_codec([[cfg.codec_bos_id]]))
                trailing = tts_pad
            else:
                parts.append(self._embed_text(input_id[:, 3:4]) + codec_input[:, -1:, :])
                trailing = mx.concatenate([self._embed_text(input_id[:, 4:-5]), tts_eos], axis=1)
        return GenerationPrompt(
            input_embeds=mx.concatenate(parts, axis=1),
            trailing_text_hidden=trailing,
            tts_pad_embed=tts_pad,
            ref_codes=ref_codes,
            text=str(text),
        )

    def _icl_prompt(self, *, text_id, ref_id, ref_codes, tts_pad, tts_eos, non_streaming_mode: bool):
        """Upstream ``generate_icl_prompt``: reference transcript + text over the reference codes."""
        cfg = self.config.talker_config
        text_embed = mx.concatenate([self._embed_text(mx.concatenate([ref_id, text_id], axis=1)), tts_eos], axis=1)
        codes = ref_codes.astype(mx.int32)  # (T, 16)
        codec_embed = self._embed_codec(codes[:, 0][None, :])
        for i in range(cfg.num_code_groups - 1):
            codec_embed = codec_embed + self.talker.code_predictor.codec_embedding[i](codes[:, i + 1][None, :])
        codec_embed = mx.concatenate([self._embed_codec([[cfg.codec_bos_id]]), codec_embed], axis=1)
        text_len, codec_len = text_embed.shape[1], codec_embed.shape[1]
        if non_streaming_mode:
            icl = text_embed + self._embed_codec([[cfg.codec_pad_id]])
            icl = mx.concatenate([icl, codec_embed + tts_pad], axis=1)
            return icl, tts_pad
        if text_len > codec_len:
            return text_embed[:, :codec_len, :] + codec_embed, text_embed[:, codec_len:, :]
        padded = mx.concatenate([text_embed, mx.broadcast_to(tts_pad, (1, codec_len - text_len, tts_pad.shape[-1]))], axis=1)
        return padded + codec_embed, tts_pad

    def custom_voice_prompt(self, text: str, *, speaker: str, language: Optional[str] = None, instruct: Optional[str] = None) -> GenerationPrompt:
        if self.model_type != "custom_voice":
            raise ValueError(f"model type {self.model_type!r} does not support custom-voice synthesis")
        name = str(speaker or "").lower()
        if name not in self.supported_speakers():
            raise ValueError(f"Unsupported speakers: [{speaker!r}]. Supported: {self.supported_speakers()}")
        if self.model_size == "0b6":
            instruct = None  # upstream: the 0.6B CustomVoice ignores instructions
        speaker_embed = self._embed_codec([[int(self.config.talker_config.spk_id[name])]])[:, 0, :]
        return self._build_prompt(
            text=text, language=language, speaker=name, speaker_embed=speaker_embed, instruct=instruct or None, non_streaming_mode=True
        )

    def voice_design_prompt(self, text: str, *, instruct: str, language: Optional[str] = None) -> GenerationPrompt:
        if self.model_type != "voice_design":
            raise ValueError(f"model type {self.model_type!r} does not support voice design")
        return self._build_prompt(text=text, language=language, speaker=None, speaker_embed=None, instruct=str(instruct), non_streaming_mode=True)

    def clone_prompt(self, text: str, *, clone: MLXClonePrompt, language: Optional[str] = None, non_streaming_mode: bool = False) -> GenerationPrompt:
        if self.model_type != "base":
            raise ValueError(f"model type {self.model_type!r} does not support voice cloning")
        return self._build_prompt(
            text=text,
            language=language,
            speaker=None,
            speaker_embed=clone.speaker_embedding,
            instruct=None,
            non_streaming_mode=non_streaming_mode,
            icl=clone,
        )

    # ------------------------------------------------------------------ reference audio (Base)

    def speaker_embedding(self, audio_24k: np.ndarray) -> mx.array:
        if self.speaker_encoder is None:
            raise RuntimeError("only Base checkpoints carry a speaker encoder")
        mels = mx.array(mel_spectrogram(audio_24k))[None]
        emb = self.speaker_encoder(mels)[0]
        mx.eval(emb)
        return emb

    def encode_reference(self, audio_24k: np.ndarray) -> mx.array:
        codes = self.codec.encode(mx.array(np.asarray(audio_24k, dtype=np.float32).reshape(-1)))
        mx.eval(codes)
        return codes

    def build_clone_prompt(self, audio: np.ndarray, sample_rate: int, *, ref_text: Optional[str], x_vector_only: bool) -> MLXClonePrompt:
        from ...audio.resample import sinc_resample_mono

        wav = np.asarray(audio, dtype=np.float32)
        if wav.ndim > 1:
            wav = wav.mean(axis=-1)
        if int(sample_rate) != self.sample_rate:
            wav = sinc_resample_mono(wav, int(sample_rate), int(self.sample_rate))
        wav = wav.astype(np.float32).reshape(-1)
        if not x_vector_only and not str(ref_text or "").strip():
            raise ValueError("ref_text is required when x_vector_only_mode=False (ICL mode)")
        ref_codes = None if x_vector_only else self.encode_reference(wav)
        return MLXClonePrompt(
            ref_codes=ref_codes,
            speaker_embedding=self.speaker_embedding(wav),
            x_vector_only_mode=bool(x_vector_only),
            icl_mode=not bool(x_vector_only),
            ref_text=ref_text,
        )

    # ------------------------------------------------------------------ generation

    def generate_frames(self, prompt: GenerationPrompt, sampling: SamplingParams, *, max_new_tokens: int) -> Iterator[mx.array]:
        """Yield one (1, 16) int32 code frame per 80 ms until EOS or ``max_new_tokens``."""
        talker = self.talker
        predictor = talker.code_predictor
        cfg = self.config.talker_config
        groups = int(cfg.num_code_groups)
        eos = int(cfg.codec_eos_token_id)
        penalty = float(sampling.repetition_penalty)
        cache = talker.make_cache()
        code_cache = predictor.make_cache()
        seen = mx.zeros((int(cfg.vocab_size),), dtype=mx.bool_)
        inputs = prompt.input_embeds
        trailing = prompt.trailing_text_hidden
        n_trailing = int(trailing.shape[1])
        for step in range(int(max_new_tokens)):
            logits, hidden = talker(inputs, cache)
            logits = logits[:, -1, :].astype(mx.float32) + self._suppress_bias
            if step < MIN_NEW_FRAMES:
                logits = logits + self._eos_bias
            if penalty != 1.0:
                penalized = mx.where(logits < 0, logits * penalty, logits / penalty)
                logits = mx.where(seen[None, :], penalized, logits)
            first = _sample(
                logits, do_sample=sampling.do_sample, temperature=sampling.temperature, top_k=sampling.top_k, top_p=sampling.top_p
            )
            is_eos = first[0, 0] == eos

            for c in code_cache:
                c.reset()
            first_embed = talker.embed_codec(first)
            x = mx.concatenate([hidden[:, -1:, :], first_embed], axis=1)
            codes = [first]
            embed_sum = first_embed
            for i in range(groups - 1):
                code_logits = predictor(x, code_cache, i)[:, -1, :].astype(mx.float32)
                code = _sample(
                    code_logits,
                    do_sample=sampling.subtalker_dosample,
                    temperature=sampling.subtalker_temperature,
                    top_k=sampling.subtalker_top_k,
                    top_p=sampling.subtalker_top_p,
                )
                codes.append(code)
                x = predictor.codec_embedding[i](code)
                embed_sum = embed_sum + x
            frame = mx.concatenate(codes, axis=1)
            text_embed = trailing[:, step : step + 1, :] if step < n_trailing else prompt.tts_pad_embed
            inputs = text_embed + embed_sum
            seen = seen | (self._vocab_ids == first[0, 0])
            mx.eval(inputs, is_eos, frame, seen)
            if is_eos.item():
                return
            yield frame

    def stream(
        self,
        prompt: GenerationPrompt,
        sampling: SamplingParams,
        *,
        max_new_tokens: int,
        chunk_frames: Optional[int] = 12,
        first_chunk_frames: Optional[int] = None,
    ) -> Iterator[np.ndarray]:
        """Yield contiguous float32 waveform chunks of one utterance.

        ``chunk_frames`` codec frames per chunk after the first (``first_chunk_frames``, which
        defaults to ``chunk_frames``); ``None``/``0`` decodes everything in one chunk at the end.
        """
        decoder = self.codec.decoder
        every = int(chunk_frames or 0)
        first_n = int(first_chunk_frames or every)
        decoder.reset_state()
        try:
            if prompt.ref_codes is not None:
                # The reference codes prime the causal decoder; their audio is upstream's "cut".
                mx.eval(decoder.step(prompt.ref_codes.astype(mx.int32).T[None]))
            pending: List[mx.array] = []
            emitted = 0
            for frame in self.generate_frames(prompt, sampling, max_new_tokens=max_new_tokens):
                pending.append(frame)
                target = first_n if emitted == 0 else every
                if every > 0 and len(pending) >= target:
                    yield self._decode_pending(decoder, pending)
                    pending = []
                    emitted += 1
            if pending:
                yield self._decode_pending(decoder, pending)
        finally:
            decoder.reset_state()

    def _decode_pending(self, decoder, pending: List[mx.array]) -> np.ndarray:
        codes = mx.concatenate(pending, axis=0).T[None]  # (1, 16, n)
        wav = decoder.step(codes)[0]
        mx.eval(wav)
        return np.asarray(wav, dtype=np.float32)

    def synthesize(self, prompt: GenerationPrompt, sampling: SamplingParams, *, max_new_tokens: int) -> np.ndarray:
        chunks = list(self.stream(prompt, sampling, max_new_tokens=max_new_tokens, chunk_frames=None))
        return np.concatenate(chunks) if chunks else np.zeros((0,), dtype=np.float32)

    # ------------------------------------------------------------------ warm-up

    def prime(self, *, max_new_tokens: int = 6) -> float:
        """Run a short generation (and, on Base checkpoints, a reference encode) so Metal kernels
        are compiled at load time rather than on the first utterance. Returns the seconds spent."""
        t0 = time.perf_counter()
        sampling = SamplingParams(do_sample=False, subtalker_dosample=False, repetition_penalty=1.0)
        text = "Hello."
        if self.model_type == "custom_voice":
            prompt = self.custom_voice_prompt(text, speaker=self.supported_speakers()[0], language="auto")
        elif self.model_type == "voice_design":
            prompt = self.voice_design_prompt(text, instruct="A calm voice.", language="auto")
        else:
            silence = np.zeros((self.sample_rate,), dtype=np.float32)
            silence[:: 240] = 0.05  # a faint click train keeps the encoder away from exact zeros
            clone = self.build_clone_prompt(silence, self.sample_rate, ref_text="Hello.", x_vector_only=False)
            prompt = self.clone_prompt(text, clone=clone, language="auto")
        for _ in self.stream(prompt, sampling, max_new_tokens=max_new_tokens, chunk_frames=3):
            pass
        return time.perf_counter() - t0

    def unload(self) -> None:
        self.talker = None  # type: ignore[assignment]
        self.codec = None  # type: ignore[assignment]
        self.speaker_encoder = None
        mx.clear_cache()
