# Derived from qwen-tts 0.1.1 (https://github.com/QwenLM/Qwen3-TTS, Apache-2.0),
# upstream file: qwen_tts/inference/qwen3_tts_model.py
# The prompt formats, tokenization, generation-kwargs merging, generate flows,
# and clone-prefix decode/cut logic are upstream-verbatim: they are the byte-exact
# contract between text and the checkpoints, and a wrong <|im_start|> produces
# plausible-sounding wrong audio rather than an error.
# Local modifications, listed exhaustively:
#   1. from_pretrained constructs the concrete vendored classes and loads weights
#      EXPLICITLY via safetensors + load_state_dict(strict=True): transformers 5.8's
#      stock loader reported success for this class while assigning nothing (random
#      init -> speech-shaped babble). No Auto* registration, no registry side effects;
#      generation defaults come from generation_config.json read as plain JSON.
#   2. URL/base64/librosa audio loading removed: reference audio is a local file
#      path (soundfile) or a (np.ndarray, sr) pair; resampling goes through
#      abstractvoice.audio.resample.sinc_resample_mono (band-limited; np.interp
#      folds >Nyquist energy straight into the speaker encoder's mel range).
#   3. get_supported_speakers()/get_supported_languages() coerce to list before
#      sorting (the model returns dict_keys, which is not JSON serializable).
#   4. Batch (list) text input is not part of AbstractVoice's contract but the
#      upstream code paths are kept verbatim to stay diffable.
"""Model-facing orchestration for Qwen3-TTS: prompt building and generate flows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from .configuration_qwen3_tts import Qwen3TTSConfig
from .modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
from .processing_qwen3_tts import Qwen3TTSProcessor

AudioLike = Union[
    str,                     # local wav path
    np.ndarray,              # waveform (requires sr)
    Tuple[np.ndarray, int],  # (waveform, sr)
]

MaybeList = Union[Any, List[Any]]


@dataclass
class VoiceClonePromptItem:
    """One sample's voice-clone prompt, aligned with
    ``Qwen3TTSForConditionalGeneration.generate(..., voice_clone_prompt=...)``."""

    ref_code: Optional[torch.Tensor]                 # (T, Q) for the 12Hz tokenizer
    ref_spk_embedding: torch.Tensor                  # (D,)
    x_vector_only_mode: bool
    icl_mode: bool
    ref_text: Optional[str] = None


def load_strict_safetensors(model, snapshot_dir: str) -> None:
    """Load a snapshot's safetensors into ``model`` with ``strict=True``.

    The whole point is loudness: transformers 5.8's stock ``from_pretrained``
    reported success for this package's composite model while assigning nothing,
    which produced speech-shaped babble from random weights. Any key drift or a
    regression to a silently-lenient loader must be a hard error. Sharded
    checkpoints (``model.safetensors.index.json``) are supported.
    """
    import json
    import os

    from safetensors.torch import load_file

    index_path = os.path.join(snapshot_dir, "model.safetensors.index.json")
    state: dict = {}
    if os.path.isfile(index_path):
        with open(index_path, "r", encoding="utf-8") as fh:
            index = json.load(fh)
        for shard in sorted(set((index.get("weight_map") or {}).values())):
            state.update(load_file(os.path.join(snapshot_dir, shard)))
    else:
        weights_path = os.path.join(snapshot_dir, "model.safetensors")
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(
                f"No model.safetensors (or index) in {snapshot_dir}; "
                "non-safetensors checkpoints are not supported by this loader."
            )
        state = load_file(weights_path)

    # assign=True adopts the checkpoint tensors (and their dtype) directly;
    # the caller performs the single intended cast afterwards.
    model.load_state_dict(state, strict=True, assign=True)


class Qwen3TTSModel:
    """Wrapper over ``Qwen3TTSForConditionalGeneration`` for the three variants:
    CustomVoice (``generate_custom_voice``), VoiceDesign (``generate_voice_design``)
    and Base (``create_voice_clone_prompt`` + ``generate_voice_clone``)."""

    def __init__(self, model: Qwen3TTSForConditionalGeneration, processor, generate_defaults: Optional[Dict[str, Any]] = None):
        self.model = model
        self.processor = processor
        self.generate_defaults = generate_defaults or {}

        self.device = getattr(model, "device", None)
        if self.device is None:
            try:
                self.device = next(model.parameters()).device
            except StopIteration:
                self.device = torch.device("cpu")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *, dtype=None, **kwargs) -> "Qwen3TTSModel":
        """Load model + processor from a local snapshot directory.

        The caller (``Qwen3TTSRuntime``) resolves the snapshot offline-first and
        passes a local directory, so nothing here reaches the network.

        Weights are loaded EXPLICITLY (safetensors + ``load_state_dict(strict=True)``)
        instead of through ``PreTrainedModel.from_pretrained``: on transformers 5.8
        the stock loader reported success for this model class — zero missing, zero
        unexpected, progress bar complete — while assigning NOTHING, leaving every
        parameter at random init. The model then produced speech-shaped babble, which
        is the worst possible failure. Strict manual loading makes any future key
        drift a hard error instead of plausible garbage. (Same pattern as
        ``audiodit/runtime._load_model_from_plain_safetensors``.)
        """
        import json
        import os

        import torch
        from safetensors.torch import load_file

        path = str(pretrained_model_name_or_path)
        if kwargs:
            raise TypeError(f"Unsupported from_pretrained kwargs for Qwen3TTSModel: {sorted(kwargs)}")

        config = Qwen3TTSConfig.from_pretrained(path)
        model = Qwen3TTSForConditionalGeneration(config)
        load_strict_safetensors(model, path)
        if dtype is not None:
            model = model.to(dtype=dtype)
        model.eval()

        # The codec lives in the snapshot's speech_tokenizer/ subfolder and pins
        # its own fp32 dtype (see codec.py).
        from .codec import Qwen3TTSCodec

        codec = Qwen3TTSCodec.from_pretrained(os.path.join(path, "speech_tokenizer"))
        model.load_speech_tokenizer(codec)

        generate_config_path = os.path.join(path, "generation_config.json")
        generate_defaults = {}
        if os.path.isfile(generate_config_path):
            with open(generate_config_path, "r", encoding="utf-8") as fh:
                generate_defaults = json.load(fh)
        model.load_generate_config(generate_defaults)

        # ProcessorMixin resolves its tokenizer class by loading the config as a
        # bare PretrainedConfig, which logs an alarming (and here meaningless)
        # "model of type `qwen3_tts` to instantiate a model of type ``" line.
        import logging

        hf_logger = logging.getLogger("transformers.configuration_utils")
        previous_level = hf_logger.level
        hf_logger.setLevel(logging.ERROR)
        try:
            processor = Qwen3TTSProcessor.from_pretrained(path)
        finally:
            hf_logger.setLevel(previous_level)
        return cls(model=model, processor=processor, generate_defaults=generate_defaults)

    # ---------------------------------------------------------------- validation

    def _supported_languages_set(self) -> Optional[set]:
        langs = getattr(self.model, "get_supported_languages", None)
        if callable(langs):
            v = langs()
            if v is None:
                return None
            return set([str(x).lower() for x in v])
        return None

    def _supported_speakers_set(self) -> Optional[set]:
        spks = getattr(self.model, "get_supported_speakers", None)
        if callable(spks):
            v = spks()
            if v is None:
                return None
            return set([str(x).lower() for x in v])
        return None

    def _validate_languages(self, languages: List[str]) -> None:
        supported = self._supported_languages_set()
        if supported is None:
            return

        bad = []
        for lang in languages:
            if lang is None:
                bad.append(lang)
                continue
            if str(lang).lower() not in supported:
                bad.append(lang)
        if bad:
            raise ValueError(f"Unsupported languages: {bad}. Supported: {sorted(supported)}")

    def _validate_speakers(self, speakers: List[Optional[str]]) -> None:
        supported = self._supported_speakers_set()
        if supported is None:
            return

        bad = []
        for spk in speakers:
            if spk is None or spk == "":
                continue
            if str(spk).lower() not in supported:
                bad.append(spk)
        if bad:
            raise ValueError(f"Unsupported speakers: {bad}. Supported: {sorted(supported)}")

    # --------------------------------------------------------------- audio input

    def _load_audio_to_np(self, x: str) -> Tuple[np.ndarray, int]:
        import soundfile as sf

        audio, sr = sf.read(x, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)
        return audio.astype(np.float32), int(sr)

    def _normalize_audio_inputs(self, audios: Union[AudioLike, List[AudioLike]]) -> List[Tuple[np.ndarray, int]]:
        """Normalize audio inputs into a list of (waveform, sr)."""
        if isinstance(audios, list):
            items = audios
        else:
            items = [audios]

        out: List[Tuple[np.ndarray, int]] = []
        for a in items:
            if isinstance(a, str):
                out.append(self._load_audio_to_np(a))
            elif isinstance(a, tuple) and len(a) == 2 and isinstance(a[0], np.ndarray):
                out.append((a[0].astype(np.float32), int(a[1])))
            elif isinstance(a, np.ndarray):
                raise ValueError("For numpy waveform input, pass a tuple (audio, sr).")
            else:
                raise TypeError(f"Unsupported audio input type: {type(a)}")
        for i, a in enumerate(out):
            if a[0].ndim > 1:
                out[i] = (np.mean(a[0], axis=-1).astype(np.float32), a[1])
        return out

    def _ensure_list(self, x: MaybeList) -> List[Any]:
        return x if isinstance(x, list) else [x]

    # ------------------------------------------------------------ prompt building
    # These strings are the byte-exact text contract with the checkpoints.

    def _build_assistant_text(self, text: str) -> str:
        return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"

    def _build_ref_text(self, text: str) -> str:
        return f"<|im_start|>assistant\n{text}<|im_end|>\n"

    def _build_instruct_text(self, instruct: str) -> str:
        return f"<|im_start|>user\n{instruct}<|im_end|>\n"

    def _tokenize_texts(self, texts: List[str]) -> List[torch.Tensor]:
        input_ids = []
        for text in texts:
            input = self.processor(text=text, return_tensors="pt", padding=True)
            input_id = input["input_ids"].to(self.device)
            input_id = input_id.unsqueeze(0) if input_id.dim() == 1 else input_id
            input_ids.append(input_id)
        return input_ids

    def _merge_generate_kwargs(
        self,
        do_sample: Optional[bool] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        subtalker_dosample: Optional[bool] = None,
        subtalker_top_k: Optional[int] = None,
        subtalker_top_p: Optional[float] = None,
        subtalker_temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """User value > generation_config.json value > hard default."""
        hard_defaults = dict(
            do_sample=True,
            top_k=50,
            top_p=1.0,
            temperature=0.9,
            repetition_penalty=1.05,
            subtalker_dosample=True,
            subtalker_top_k=50,
            subtalker_top_p=1.0,
            subtalker_temperature=0.9,
            max_new_tokens=2048,
        )

        def pick(name: str, user_val: Any) -> Any:
            if user_val is not None:
                return user_val
            if name in self.generate_defaults:
                return self.generate_defaults[name]
            return hard_defaults[name]

        merged = dict(kwargs)
        merged.update(
            do_sample=pick("do_sample", do_sample),
            top_k=pick("top_k", top_k),
            top_p=pick("top_p", top_p),
            temperature=pick("temperature", temperature),
            repetition_penalty=pick("repetition_penalty", repetition_penalty),
            subtalker_dosample=pick("subtalker_dosample", subtalker_dosample),
            subtalker_top_k=pick("subtalker_top_k", subtalker_top_k),
            subtalker_top_p=pick("subtalker_top_p", subtalker_top_p),
            subtalker_temperature=pick("subtalker_temperature", subtalker_temperature),
            max_new_tokens=pick("max_new_tokens", max_new_tokens),
        )
        return merged

    # -------------------------------------------------------------- voice clone

    @torch.inference_mode()
    def create_voice_clone_prompt(
        self,
        ref_audio: Union[AudioLike, List[AudioLike]],
        ref_text: Optional[Union[str, List[Optional[str]]]] = None,
        x_vector_only_mode: Union[bool, List[bool]] = False,
    ) -> List[VoiceClonePromptItem]:
        """Build voice-clone prompt items from reference audio using the Base model.

        ``x_vector_only_mode=False`` (ICL) requires ``ref_text``; the caller owns
        that decision — this method never infers a mode (ADR 0003).
        """
        if self.model.tts_model_type != "base":
            raise ValueError(
                f"model with \ntokenizer_type: {self.model.tokenizer_type}\n"
                f"tts_model_size: {self.model.tts_model_size}\n"
                f"tts_model_type: {self.model.tts_model_type}\n"
                "does not support create_voice_clone_prompt, Please check Model Card or Readme for more details."
            )

        from ..audio.resample import sinc_resample_mono

        ref_audio_list = self._ensure_list(ref_audio)
        ref_text_list = self._ensure_list(ref_text) if isinstance(ref_text, list) else ([ref_text] * len(ref_audio_list))
        xvec_list = self._ensure_list(x_vector_only_mode) if isinstance(x_vector_only_mode, list) else ([x_vector_only_mode] * len(ref_audio_list))

        if len(ref_text_list) != len(ref_audio_list) or len(xvec_list) != len(ref_audio_list):
            raise ValueError(
                f"Batch size mismatch: ref_audio={len(ref_audio_list)}, ref_text={len(ref_text_list)}, x_vector_only_mode={len(xvec_list)}"
            )

        normalized = self._normalize_audio_inputs(ref_audio_list)

        ref_wavs_for_code: List[np.ndarray] = []
        ref_sr_for_code: List[int] = []
        for wav, sr in normalized:
            ref_wavs_for_code.append(wav)
            ref_sr_for_code.append(sr)

        if len(set(ref_sr_for_code)) == 1:
            enc = self.model.speech_tokenizer.encode(ref_wavs_for_code, sr=ref_sr_for_code[0])
            ref_codes = enc.audio_codes
        else:
            ref_codes = []
            for wav, sr in normalized:
                ref_codes.append(self.model.speech_tokenizer.encode(wav, sr=sr).audio_codes[0])

        items: List[VoiceClonePromptItem] = []
        for i, ((wav, sr), code, rtext, xvec_only) in enumerate(zip(normalized, ref_codes, ref_text_list, xvec_list)):
            if not xvec_only:
                if rtext is None or rtext == "":
                    raise ValueError(f"ref_text is required when x_vector_only_mode=False (ICL mode). Bad index={i}")

            wav_resample = wav
            if sr != self.model.speaker_encoder_sample_rate:
                wav_resample = sinc_resample_mono(
                    wav_resample.astype(np.float32), int(sr), int(self.model.speaker_encoder_sample_rate)
                )

            spk_emb = self.model.extract_speaker_embedding(audio=wav_resample,
                                                           sr=self.model.speaker_encoder_sample_rate)

            items.append(
                VoiceClonePromptItem(
                    ref_code=None if xvec_only else code,
                    ref_spk_embedding=spk_emb,
                    x_vector_only_mode=bool(xvec_only),
                    icl_mode=bool(not xvec_only),
                    ref_text=rtext,
                )
            )
        return items

    def _prompt_items_to_voice_clone_prompt(self, items: List[VoiceClonePromptItem]) -> Dict[str, Any]:
        return dict(
            ref_code=[it.ref_code for it in items],
            ref_spk_embedding=[it.ref_spk_embedding for it in items],
            x_vector_only_mode=[it.x_vector_only_mode for it in items],
            icl_mode=[it.icl_mode for it in items],
        )

    @torch.no_grad()
    def generate_voice_clone(
        self,
        text: Union[str, List[str]],
        language: Union[str, List[str]] = None,
        ref_audio: Optional[Union[AudioLike, List[AudioLike]]] = None,
        ref_text: Optional[Union[str, List[Optional[str]]]] = None,
        x_vector_only_mode: Union[bool, List[bool]] = False,
        voice_clone_prompt: Optional[Union[Dict[str, Any], List[VoiceClonePromptItem]]] = None,
        non_streaming_mode: bool = False,
        **kwargs,
    ) -> Tuple[List[np.ndarray], int]:
        """Voice-clone speech with the Base model; see upstream for parameters."""
        if self.model.tts_model_type != "base":
            raise ValueError(
                f"model with \ntokenizer_type: {self.model.tokenizer_type}\n"
                f"tts_model_size: {self.model.tts_model_size}\n"
                f"tts_model_type: {self.model.tts_model_type}\n"
                "does not support generate_voice_clone, Please check Model Card or Readme for more details."
            )

        texts = self._ensure_list(text)
        languages = self._ensure_list(language) if isinstance(language, list) else ([language] * len(texts) if language is not None else ["Auto"] * len(texts))
        if len(languages) == 1 and len(texts) > 1:
            languages = languages * len(texts)
        if len(texts) != len(languages):
            raise ValueError(f"Batch size mismatch: text={len(texts)}, language={len(languages)}")

        self._validate_languages(languages)

        if voice_clone_prompt is None:
            if ref_audio is None:
                raise ValueError("Either `voice_clone_prompt` or `ref_audio` must be provided.")
            prompt_items = self.create_voice_clone_prompt(ref_audio=ref_audio, ref_text=ref_text, x_vector_only_mode=x_vector_only_mode)
            if len(prompt_items) == 1 and len(texts) > 1:
                prompt_items = prompt_items * len(texts)
            if len(prompt_items) != len(texts):
                raise ValueError(f"Batch size mismatch: prompt={len(prompt_items)}, text={len(texts)}")
            voice_clone_prompt_dict = self._prompt_items_to_voice_clone_prompt(prompt_items)
            ref_texts_for_ids = [it.ref_text for it in prompt_items]
        else:
            if isinstance(voice_clone_prompt, list):
                prompt_items = voice_clone_prompt
                if len(prompt_items) == 1 and len(texts) > 1:
                    prompt_items = prompt_items * len(texts)
                if len(prompt_items) != len(texts):
                    raise ValueError(f"Batch size mismatch: prompt={len(prompt_items)}, text={len(texts)}")
                voice_clone_prompt_dict = self._prompt_items_to_voice_clone_prompt(prompt_items)
                ref_texts_for_ids = [it.ref_text for it in prompt_items]
            else:
                voice_clone_prompt_dict = voice_clone_prompt
                ref_texts_for_ids = None

        input_texts = [self._build_assistant_text(t) for t in texts]
        input_ids = self._tokenize_texts(input_texts)

        ref_ids = None
        if ref_texts_for_ids is not None:
            ref_ids = []
            for i, rt in enumerate(ref_texts_for_ids):
                if rt is None or rt == "":
                    ref_ids.append(None)
                else:
                    ref_tok = self._tokenize_texts([self._build_ref_text(rt)])[0]
                    ref_ids.append(ref_tok)

        gen_kwargs = self._merge_generate_kwargs(**kwargs)

        talker_codes_list, _ = self.model.generate(
            input_ids=input_ids,
            ref_ids=ref_ids,
            voice_clone_prompt=voice_clone_prompt_dict,
            languages=languages,
            non_streaming_mode=non_streaming_mode,
            **gen_kwargs,
        )

        codes_for_decode = []
        for i, codes in enumerate(talker_codes_list):
            ref_code_list = voice_clone_prompt_dict.get("ref_code", None)
            if ref_code_list is not None and ref_code_list[i] is not None:
                codes_for_decode.append(torch.cat([ref_code_list[i].to(codes.device), codes], dim=0))
            else:
                codes_for_decode.append(codes)

        wavs_all, fs = self.model.speech_tokenizer.decode([{"audio_codes": c} for c in codes_for_decode])

        wavs_out: List[np.ndarray] = []
        for i, wav in enumerate(wavs_all):
            ref_code_list = voice_clone_prompt_dict.get("ref_code", None)
            if ref_code_list is not None and ref_code_list[i] is not None:
                ref_len = int(ref_code_list[i].shape[0])
                total_len = int(codes_for_decode[i].shape[0])
                cut = int(ref_len / max(total_len, 1) * wav.shape[0])
                wavs_out.append(wav[cut:])
            else:
                wavs_out.append(wav)

        return wavs_out, fs

    # -------------------------------------------------------------- voice design

    @torch.no_grad()
    def generate_voice_design(
        self,
        text: Union[str, List[str]],
        instruct: Union[str, List[str]],
        language: Union[str, List[str]] = None,
        non_streaming_mode: bool = True,
        **kwargs,
    ) -> Tuple[List[np.ndarray], int]:
        """Generate speech with the VoiceDesign model from a style instruction."""
        if self.model.tts_model_type != "voice_design":
            raise ValueError(
                f"model with \ntokenizer_type: {self.model.tokenizer_type}\n"
                f"tts_model_size: {self.model.tts_model_size}\n"
                f"tts_model_type: {self.model.tts_model_type}\n"
                "does not support generate_voice_design, Please check Model Card or Readme for more details."
            )

        texts = self._ensure_list(text)
        languages = self._ensure_list(language) if isinstance(language, list) else ([language] * len(texts) if language is not None else ["Auto"] * len(texts))
        instructs = self._ensure_list(instruct)

        if len(languages) == 1 and len(texts) > 1:
            languages = languages * len(texts)
        if len(instructs) == 1 and len(texts) > 1:
            instructs = instructs * len(texts)

        if not (len(texts) == len(languages) == len(instructs)):
            raise ValueError(f"Batch size mismatch: text={len(texts)}, language={len(languages)}, instruct={len(instructs)}")

        self._validate_languages(languages)

        input_ids = self._tokenize_texts([self._build_assistant_text(t) for t in texts])

        instruct_ids: List[Optional[torch.Tensor]] = []
        for ins in instructs:
            if ins is None or ins == "":
                instruct_ids.append(None)
            else:
                instruct_ids.append(self._tokenize_texts([self._build_instruct_text(ins)])[0])

        gen_kwargs = self._merge_generate_kwargs(**kwargs)

        talker_codes_list, _ = self.model.generate(
            input_ids=input_ids,
            instruct_ids=instruct_ids,
            languages=languages,
            non_streaming_mode=non_streaming_mode,
            **gen_kwargs,
        )

        wavs, fs = self.model.speech_tokenizer.decode([{"audio_codes": c} for c in talker_codes_list])
        return wavs, fs

    # -------------------------------------------------------------- custom voice

    @torch.no_grad()
    def generate_custom_voice(
        self,
        text: Union[str, List[str]],
        speaker: Union[str, List[str]],
        language: Union[str, List[str]] = None,
        instruct: Optional[Union[str, List[str]]] = None,
        non_streaming_mode: bool = True,
        **kwargs,
    ) -> Tuple[List[np.ndarray], int]:
        """Generate speech with the CustomVoice model using a preset speaker."""
        if self.model.tts_model_type != "custom_voice":
            raise ValueError(
                f"model with \ntokenizer_type: {self.model.tokenizer_type}\n"
                f"tts_model_size: {self.model.tts_model_size}\n"
                f"tts_model_type: {self.model.tts_model_type}\n"
                "does not support generate_custom_voice, Please check Model Card or Readme for more details."
            )

        texts = self._ensure_list(text)
        languages = self._ensure_list(language) if isinstance(language, list) else ([language] * len(texts) if language is not None else ["Auto"] * len(texts))
        speakers = self._ensure_list(speaker)
        if self.model.tts_model_size in "0b6": # for 0b6 model, instruct is not supported
            instruct = None
        instructs = self._ensure_list(instruct) if isinstance(instruct, list) else ([instruct] * len(texts) if instruct is not None else [""] * len(texts))

        if len(languages) == 1 and len(texts) > 1:
            languages = languages * len(texts)
        if len(speakers) == 1 and len(texts) > 1:
            speakers = speakers * len(texts)
        if len(instructs) == 1 and len(texts) > 1:
            instructs = instructs * len(texts)

        if not (len(texts) == len(languages) == len(speakers) == len(instructs)):
            raise ValueError(
                f"Batch size mismatch: text={len(texts)}, language={len(languages)}, speaker={len(speakers)}, instruct={len(instructs)}"
            )

        self._validate_languages(languages)
        self._validate_speakers(speakers)

        input_ids = self._tokenize_texts([self._build_assistant_text(t) for t in texts])

        instruct_ids: List[Optional[torch.Tensor]] = []
        for ins in instructs:
            if ins is None or ins == "":
                instruct_ids.append(None)
            else:
                instruct_ids.append(self._tokenize_texts([self._build_instruct_text(ins)])[0])

        gen_kwargs = self._merge_generate_kwargs(**kwargs)

        talker_codes_list, _ = self.model.generate(
            input_ids=input_ids,
            instruct_ids=instruct_ids,
            languages=languages,
            speakers=speakers,
            non_streaming_mode=non_streaming_mode,
            **gen_kwargs,
        )

        wavs, fs = self.model.speech_tokenizer.decode([{"audio_codes": c} for c in talker_codes_list])
        return wavs, fs

    # ------------------------------------------------------------------ metadata

    def get_supported_speakers(self) -> Optional[List[str]]:
        """Sorted supported speaker names (lowercased), or None if unconstrained."""
        supported = self._supported_speakers_set()
        if supported is None:
            return None
        return sorted(supported)

    def get_supported_languages(self) -> Optional[List[str]]:
        """Sorted supported language names (lowercased), or None if unconstrained."""
        supported = self._supported_languages_set()
        if supported is None:
            return None
        return sorted(supported)
