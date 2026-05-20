"""Transformers-based local ASR (Hugging Face) adapter.

This adapter enables selecting Hugging Face ASR models such as:
- openai/whisper-large-v3
- openai/whisper-large-v3-turbo
- Qwen/Qwen3-ASR-1.7B

Design goals:
- Optional dependency: only imported/loaded when explicitly selected.
- Offline-first: when allow_downloads=False, never pulls from the network.
"""

from __future__ import annotations

import io
import os
import warnings
from typing import Any, Dict, Optional

import numpy as np

from ..audio.resample import linear_resample_mono
from ..compute import looks_like_torch_device_error, resolve_torch_runtime
from .base import STTAdapter


def _env_bool(key: str, default: bool = False) -> bool:
    raw = os.environ.get(str(key), None)
    if raw is None:
        return bool(default)
    val = str(raw).strip().lower()
    if not val:
        return bool(default)
    return val in {"1", "true", "yes", "y", "on"}


def _safe_float32_mono(audio: Any) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim == 2:
        # (n, c) -> mono
        x = np.mean(x, axis=1).astype(np.float32)
    return x.reshape(-1).astype(np.float32)


def _is_qwen3_asr_model_id(model_id: str) -> bool:
    text = str(model_id or "").strip().lower()
    if not text:
        return False
    return "qwen3-asr" in text or "qwen3_asr" in text


_QWEN3_ASR_LANG_BY_CODE: dict[str, str] = {
    "zh": "Chinese",
    "en": "English",
    "yue": "Cantonese",
    "ar": "Arabic",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "id": "Indonesian",
    "it": "Italian",
    "ko": "Korean",
    "ru": "Russian",
    "th": "Thai",
    "vi": "Vietnamese",
    "ja": "Japanese",
    "tr": "Turkish",
    "hi": "Hindi",
    "ms": "Malay",
    "nl": "Dutch",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "pl": "Polish",
    "cs": "Czech",
    "fil": "Filipino",
    "fa": "Persian",
    "el": "Greek",
    "ro": "Romanian",
    "hu": "Hungarian",
    "mk": "Macedonian",
}


def _qwen3_asr_language_hint(language: str | None) -> str | None:
    if not language:
        return None
    raw = str(language).strip()
    if not raw:
        return None
    key = raw.lower().replace("_", "-")
    # Accept both ISO-ish codes and already-canonical names (e.g. "English").
    return _QWEN3_ASR_LANG_BY_CODE.get(key, raw)


def _parse_qwen3_asr_text(raw: Any, *, user_language: str | None = None) -> str:
    if raw is None:
        return ""
    s = str(raw).strip()
    if not s:
        return ""
    if user_language:
        # When forcing language, the model is prompted to emit text-only.
        return s
    tag = "<asr_text>"
    if tag not in s:
        return s
    meta, text = s.split(tag, 1)
    if "language none" in meta.lower() and not str(text or "").strip():
        return ""
    return str(text or "").strip()


class TransformersASRAdapter(STTAdapter):
    """Local STT adapter backed by Hugging Face Transformers pipelines."""

    ENGINE_ID = "transformers-asr"

    # A small curated set of well-known HF model ids we want to expose explicitly.
    # Users may still pass any HF model id or local path via `model_id`.
    KNOWN_MODELS: dict[str, str] = {
        "openai/whisper-large-v3": "Whisper large-v3 (Transformers)",
        "openai/whisper-large-v3-turbo": "Whisper large-v3-turbo (Transformers)",
        "Qwen/Qwen3-ASR-1.7B": "Qwen3 ASR 1.7B (Transformers)",
    }

    # Aliases for convenience and parity with other adapters.
    _MODEL_ALIASES: dict[str, str] = {
        "whisper-large-v3": "openai/whisper-large-v3",
        "whisper-large-v3-turbo": "openai/whisper-large-v3-turbo",
        "qwen3-asr-1.7b": "Qwen/Qwen3-ASR-1.7B",
    }

    @classmethod
    def selectable_model_ids(cls) -> list[str]:
        """Return canonical ids plus convenience aliases for discovery/UI."""
        return list(dict.fromkeys([*cls.KNOWN_MODELS.keys(), *cls._MODEL_ALIASES.keys()]))

    # Keep a conservative list for UI/default validation; many HF models support more.
    LANGUAGES = [
        "en",
        "fr",
        "de",
        "es",
        "ru",
        "zh",
        "it",
        "pt",
        "ja",
        "ko",
        "ar",
        "hi",
    ]

    def __init__(
        self,
        *,
        model_id: str,
        device: str = "auto",
        dtype: str | None = None,
        allow_downloads: bool = True,
        trust_remote_code: bool | None = None,
    ) -> None:
        self.engine_id = self.ENGINE_ID
        self.provider = self.ENGINE_ID
        self._available = False
        self._pipeline = None
        self._qwen3_model = None
        self._qwen3_processor = None
        self._target_sample_rate = 16000

        raw = str(model_id or "").strip()
        if not raw:
            raise ValueError("model_id is required for transformers-asr")
        alias_key = raw.strip().lower()
        self._model_id = self._MODEL_ALIASES.get(alias_key, raw)
        self.model_id = self._model_id

        self._device_pref = str(device or "auto").strip().lower() or "auto"
        self._dtype_pref = str(dtype).strip().lower() if dtype else None
        self._allow_downloads = bool(allow_downloads)
        self._trust_remote_code = (
            bool(trust_remote_code)
            if trust_remote_code is not None
            else _env_bool("ABSTRACTVOICE_HF_TRUST_REMOTE_CODE", False)
        )
        self._use_qwen3_asr = _is_qwen3_asr_model_id(self._model_id)
        if self._use_qwen3_asr:
            # Qwen3-ASR runs through our vendored backend and never requires
            # executing Hugging Face repo Python code.
            self._trust_remote_code = False

        self._current_language: str | None = None
        self._resolved_device: str | None = None
        self._resolved_dtype: str | None = None
        self._used_fallback = False
        self._fallback_reason: str | None = None
        self._load_error: str | None = None

        # Best-effort eager load so `is_available()` is meaningful right away.
        try:
            self._ensure_loaded()
        except Exception as e:
            # Offline-first: missing cached weights is a normal outcome when allow_downloads=False.
            self._load_error = str(e)
            self._available = False

    def _resolve_runtime(self):
        return resolve_torch_runtime(
            device=str(self._device_pref or "auto"),
            dtype_name=self._dtype_pref,
            allow_cpu_fallback=str(self._device_pref or "auto") == "auto",
        )

    def _capture_model_runtime(self, model: Any) -> None:
        if model is None or not hasattr(model, "parameters"):
            return
        try:
            first_param = next(iter(model.parameters()), None)
        except Exception:
            first_param = None
        if first_param is None:
            return
        try:
            if hasattr(first_param, "device"):
                self._resolved_device = str(first_param.device)
        except Exception:
            pass
        try:
            if hasattr(first_param, "dtype"):
                self._resolved_dtype = str(first_param.dtype).replace("torch.", "")
        except Exception:
            pass

    def runtime_info(self) -> Dict[str, Any]:
        return {
            "requested_device": self._device_pref,
            "resolved_device": self._resolved_device,
            "requested_dtype": self._dtype_pref,
            "resolved_dtype": self._resolved_dtype,
            "used_fallback": bool(self._used_fallback),
            "fallback_reason": self._fallback_reason,
            "load_error": self._load_error,
        }

    def get_unavailable_reason(self) -> str | None:
        return self._load_error

    def _ensure_loaded(self) -> None:
        if self._pipeline is not None or self._qwen3_model is not None:
            return

        # Keep interactive UX quiet by default.
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TRANSFORMERS_NO_TQDM", "1")
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

        try:
            import torch
            from transformers import pipeline
            from transformers.utils import logging as _tf_logging
        except Exception as e:
            raise RuntimeError(
                "Transformers ASR requires optional dependencies.\n"
                "Install with:\n"
                "  pip install \"abstractvoice[stt-hf]\"\n"
                "  pip install \"abstractvoice[apple]\"  # Apple profile\n"
                "  pip install \"abstractvoice[gpu]\"    # GPU profile"
            ) from e

        try:
            _tf_logging.disable_progress_bar()
        except Exception:
            pass

        local_only = not bool(self._allow_downloads)
        old_hf_offline = os.environ.get("HF_HUB_OFFLINE")
        old_tf_offline = os.environ.get("TRANSFORMERS_OFFLINE")
        old_disable_pb = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
        if local_only:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

        self._available = False
        try:
            runtime = self._resolve_runtime()
            self._resolved_device = str(runtime.resolved_device)
            self._resolved_dtype = str(runtime.resolved_dtype_name)
            self._used_fallback = bool(runtime.used_fallback)
            self._fallback_reason = runtime.fallback_reason
            if runtime.used_fallback and runtime.fallback_reason:
                warnings.warn(runtime.fallback_reason)
            torch_device = torch.device(str(runtime.resolved_device))
            torch_dtype = runtime.torch_dtype
            self._load_backend(
                pipeline=pipeline,
                torch_device=torch_device,
                torch_dtype=torch_dtype,
                local_only=bool(local_only),
            )
        except Exception as e:
            runtime = locals().get("runtime")
            resolved_device = str(getattr(runtime, "resolved_device", "") or "").lower()
            if not resolved_device:
                self._load_error = str(e)
                raise
            if (
                str(self._device_pref or "auto") == "auto"
                and resolved_device
                and resolved_device != "cpu"
                and looks_like_torch_device_error(e, attempted_device=resolved_device)
            ):
                try:
                    cpu_runtime = resolve_torch_runtime(
                        device="cpu",
                        dtype_name="float32",
                        allow_cpu_fallback=False,
                    )
                    retry_reason = (
                        f"Falling back to CPU because transformers-asr load failed on '{resolved_device}': "
                        f"{type(e).__name__}: {e}"
                    )
                    warnings.warn(retry_reason)
                    self._resolved_device = str(cpu_runtime.resolved_device)
                    self._resolved_dtype = str(cpu_runtime.resolved_dtype_name)
                    self._used_fallback = True
                    self._fallback_reason = retry_reason
                    self._load_backend(
                        pipeline=pipeline,
                        torch_device=torch.device(str(cpu_runtime.resolved_device)),
                        torch_dtype=cpu_runtime.torch_dtype,
                        local_only=bool(local_only),
                    )
                    self._load_error = None
                    return
                except Exception:
                    pass

            if local_only:
                self._load_error = (
                    "Transformers ASR model is not available locally and downloads are disabled.\n"
                    "Fix options:\n"
                    "  - Enable downloads: VoiceManager(..., allow_downloads=True)\n"
                    "  - Or prefetch explicitly: abstractvoice-prefetch --stt-hf <model_id>\n"
                    f"Model: {self._model_id}"
                )
                raise RuntimeError(
                    self._load_error
                ) from e
            self._load_error = str(e)
            raise
        finally:
            if local_only:
                if old_hf_offline is None:
                    os.environ.pop("HF_HUB_OFFLINE", None)
                else:
                    os.environ["HF_HUB_OFFLINE"] = old_hf_offline
                if old_tf_offline is None:
                    os.environ.pop("TRANSFORMERS_OFFLINE", None)
                else:
                    os.environ["TRANSFORMERS_OFFLINE"] = old_tf_offline
                if old_disable_pb is None:
                    os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
                else:
                    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = old_disable_pb

        self._load_error = None
        if self._pipeline is not None:
            try:
                sr = int(getattr(getattr(self._pipeline, "feature_extractor", None), "sampling_rate", 16000))
                self._target_sample_rate = sr if sr > 0 else 16000
            except Exception:
                self._target_sample_rate = 16000

        self._available = True

    def _load_backend(self, *, pipeline, torch_device: Any, torch_dtype: Any, local_only: bool) -> None:
        self._pipeline = None
        self._qwen3_model = None
        self._qwen3_processor = None

        if self._use_qwen3_asr:
            self._ensure_loaded_qwen3_asr(
                torch_device=torch_device,
                torch_dtype=torch_dtype,
                local_only=bool(local_only),
            )
            return

        # Avoid noisy HF Hub token warnings when operating in local-only mode.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"^Warning: You are sending unauthenticated requests to the HF Hub\\..*",
            )
            warnings.filterwarnings(
                "ignore",
                message=r"^You are sending unauthenticated requests to the HF Hub\\..*",
            )
            pipe = pipeline(
                "automatic-speech-recognition",
                model=self._model_id,
                device=torch_device,
                dtype=torch_dtype if torch_dtype is not None else "auto",
                trust_remote_code=bool(self._trust_remote_code),
                model_kwargs={"local_files_only": bool(local_only)},
            )
        self._pipeline = pipe
        self._capture_model_runtime(getattr(pipe, "model", None))

    def _ensure_loaded_qwen3_asr(self, *, torch_device: Any, torch_dtype: Any, local_only: bool) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except Exception as e:
            raise RuntimeError(
                "Qwen3-ASR requires Transformers + Torch optional dependencies.\n"
                "Install with:\n"
                "  pip install \"abstractvoice[stt-hf]\"\n"
                "  pip install \"abstractvoice[apple]\"  # Apple profile\n"
                "  pip install \"abstractvoice[gpu]\"    # GPU profile"
            ) from e

        from ..qwen3_asr import register_transformers_qwen3_asr

        register_transformers_qwen3_asr()

        model_kwargs: dict[str, Any] = {
            "local_files_only": bool(local_only),
            "trust_remote_code": False,
        }
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype

        # Best-effort device placement: avoid device_map magic by default, but allow it
        # when the resolved device is clearly GPU-like.
        resolved_device = str(getattr(torch_device, "type", "") or torch_device).lower()
        if resolved_device.startswith("cuda"):
            model_kwargs["device_map"] = resolved_device
        elif resolved_device.startswith("mps"):
            # MPS does not support accelerate-style device maps reliably; load then move.
            model_kwargs.pop("device_map", None)

        model = AutoModel.from_pretrained(self._model_id, **model_kwargs)
        try:
            if resolved_device.startswith("cuda") or resolved_device.startswith("mps"):
                model = model.to(torch_device)
        except Exception as e:
            if str(self._device_pref or "auto") == "auto":
                raise
            warning = (
                f"Failed to move Qwen3-ASR model to requested device '{resolved_device}': {e}. "
                "Continuing on the model's current load device."
            )
            self._used_fallback = True
            self._fallback_reason = warning
            warnings.warn(warning, RuntimeWarning)

        # `fix_mistral_regex` is a Qwen3-ASR processor option; ignore if unsupported.
        processor_kwargs: dict[str, Any] = {"local_files_only": bool(local_only), "trust_remote_code": False}
        try:
            processor_kwargs["fix_mistral_regex"] = True
        except Exception:
            pass
        processor = AutoProcessor.from_pretrained(self._model_id, **processor_kwargs)

        self._qwen3_model = model
        self._qwen3_processor = processor
        self._capture_model_runtime(model)
        self._target_sample_rate = int(getattr(getattr(processor, "feature_extractor", None), "sampling_rate", 16000) or 16000)
        if self._target_sample_rate <= 0:
            self._target_sample_rate = 16000
        self._available = True

    def _transcribe_array(self, audio: np.ndarray, sample_rate: int, language: str | None) -> str:
        self._ensure_loaded()
        if self._use_qwen3_asr:
            if self._qwen3_model is None or self._qwen3_processor is None:
                raise RuntimeError("Qwen3-ASR model is not available")
        else:
            if self._pipeline is None:
                raise RuntimeError("Transformers ASR pipeline is not available")

        x = _safe_float32_mono(audio)
        sr = int(sample_rate)
        target_sr = int(self._target_sample_rate or 16000)
        if sr and target_sr and sr != target_sr:
            x = linear_resample_mono(x, sr, target_sr)
            sr = target_sr

        if self._use_qwen3_asr:
            model = self._qwen3_model
            processor = self._qwen3_processor
            if model is None or processor is None:
                raise RuntimeError("Qwen3-ASR is not initialized")

            forced_language = _qwen3_asr_language_hint(language)

            # Qwen3-ASR expects a chat-style prompt with audio placeholders.
            messages = [
                {"role": "system", "content": ""},
                {"role": "user", "content": [{"type": "audio", "audio": ""}]},
            ]
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            user_lang = None
            if forced_language:
                user_lang = forced_language
                prompt = prompt + f"language {forced_language}<asr_text>"

            inputs = processor(text=[prompt], audio=[x], return_tensors="pt", padding=True)
            try:
                inputs = inputs.to(model.device).to(model.dtype)
            except Exception:
                try:
                    inputs = inputs.to(model.device)
                except Exception:
                    pass

            out = model.generate(**inputs, max_new_tokens=512)
            sequences = getattr(out, "sequences", out)
            try:
                prompt_len = int(inputs["input_ids"].shape[1])
                sequences = sequences[:, prompt_len:]
            except Exception:
                pass

            decoded = processor.batch_decode(
                sequences,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            raw_text = str(decoded[0] if decoded else "").strip()
            return _parse_qwen3_asr_text(raw_text, user_language=user_lang)

        # Whisper language/task selection uses forced decoder ids.
        generate_kwargs: dict[str, Any] = {}
        if language:
            tok = getattr(self._pipeline, "tokenizer", None)
            get_prompt = getattr(tok, "get_decoder_prompt_ids", None)
            if callable(get_prompt):
                try:
                    forced = get_prompt(task="transcribe", language=str(language).strip().lower(), no_timestamps=True)
                    if forced:
                        generate_kwargs["forced_decoder_ids"] = forced
                except Exception:
                    # Best-effort: not all models/tokenizers support this.
                    pass

        # Transformers ASR pipeline assumes input sampling rate equals feature_extractor.sampling_rate.
        result = self._pipeline(x, generate_kwargs=generate_kwargs) if generate_kwargs else self._pipeline(x)

        if isinstance(result, dict):
            text = result.get("text")
            return str(text or "").strip()
        if isinstance(result, list) and result:
            first = result[0] if isinstance(result[0], dict) else {}
            return str(first.get("text") or "").strip()
        return str(result or "").strip()

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        *,
        hotwords: Optional[str] = None,
        initial_prompt: Optional[str] = None,
        condition_on_previous_text: bool = True,
    ) -> str:
        _ = hotwords, initial_prompt, condition_on_previous_text
        if not self.is_available():
            self._ensure_loaded()

        try:
            import soundfile as sf
        except Exception as e:
            raise RuntimeError(
                "Transformers ASR requires soundfile to load audio files.\n"
                "Install with: pip install \"abstractvoice[stt-hf]\""
            ) from e

        audio, sr = sf.read(str(audio_path), always_2d=True, dtype="float32")
        mono = _safe_float32_mono(audio)
        return self._transcribe_array(mono, int(sr), language)

    def transcribe_from_bytes(
        self,
        audio_bytes: bytes,
        language: Optional[str] = None,
        *,
        hotwords: Optional[str] = None,
        initial_prompt: Optional[str] = None,
        condition_on_previous_text: bool = True,
    ) -> str:
        _ = hotwords, initial_prompt, condition_on_previous_text
        if not self.is_available():
            self._ensure_loaded()

        try:
            import soundfile as sf
        except Exception as e:
            raise RuntimeError(
                "Transformers ASR requires soundfile to load audio bytes.\n"
                "Install with: pip install \"abstractvoice[stt-hf]\""
            ) from e

        with sf.SoundFile(io.BytesIO(bytes(audio_bytes))) as f:
            sr = int(f.samplerate)
            audio = f.read(dtype="float32", always_2d=True)
        mono = _safe_float32_mono(audio)
        return self._transcribe_array(mono, int(sr), language)

    def transcribe_from_array(
        self,
        audio_array: np.ndarray,
        sample_rate: int,
        language: Optional[str] = None,
        *,
        hotwords: Optional[str] = None,
        initial_prompt: Optional[str] = None,
        condition_on_previous_text: bool = True,
    ) -> str:
        _ = hotwords, initial_prompt, condition_on_previous_text
        if not self.is_available():
            self._ensure_loaded()
        return self._transcribe_array(np.asarray(audio_array), int(sample_rate), language)

    def set_language(self, language: str) -> bool:
        text = str(language or "").strip().lower()
        if not text:
            return False
        self._current_language = text
        return True

    def get_supported_languages(self) -> list[str]:
        return list(self.LANGUAGES)

    def is_available(self) -> bool:
        if not bool(self._available):
            return False
        if self._use_qwen3_asr:
            return self._qwen3_model is not None and self._qwen3_processor is not None
        return self._pipeline is not None

    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        info.update(
            {
                "engine_id": self.ENGINE_ID,
                "provider": self.ENGINE_ID,
                "model_id": self._model_id,
                "target_sample_rate": int(self._target_sample_rate or 0),
                "allow_downloads": bool(self._allow_downloads),
                "trust_remote_code": bool(self._trust_remote_code) if not self._use_qwen3_asr else False,
                "backend": "qwen3-asr" if self._use_qwen3_asr else "transformers-pipeline",
                "runtime": self.runtime_info(),
            }
        )
        return info
