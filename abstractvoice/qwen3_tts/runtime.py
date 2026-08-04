"""Qwen3-TTS runtime wrapper (AbstractVoice-owned).

Maps AbstractVoice concepts onto the vendored Qwen3-TTS core: offline-first
snapshot resolution, the shared torch device/dtype policy (ADR 0005), model
variant gating (custom_voice / base / voice_design), sampling presets, and a
serialization lock so concurrent synthesis cannot interleave generation state.

Everything model-facing (prompt formats, generate flows) lives in
``orchestration.py``, upstream-verbatim. Everything policy-facing lives here.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from ..compute import resolve_torch_runtime

DEFAULT_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"

# The five 12Hz repos this integration understands. Others load too — the gate
# is `tts_model_type` read from the snapshot config, not this list.
KNOWN_MODEL_IDS = (
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
)

# Codec frame rate: 12.5 codes/second of audio. max_new_tokens derives from the
# text length instead of riding generation_config.json's 8192 (= ~11 minutes of
# audio per utterance -- a runaway-generation budget, not a sane cap).
_CODES_PER_SECOND = 12.5
# The SLOWEST measured speaking rate across the preset voices: dylan renders
# English at 4.8 chars/s (and CJK syllabary sits near 5 chars/s), so 4.0 with
# the 1.5x factor leaves ~1.9x headroom over the slowest real clip. EOS ends
# normal generation long before the cap; a tight "fit" here silently truncated
# half a sentence for slow presets, which is worse than a loose cap.
_MIN_CHARS_PER_SECOND = 4.0


def estimate_max_new_tokens(text: str, *, floor: int = 96, ceiling: int = 2048) -> int:
    """RUNAWAY cap for one utterance's codec tokens, derived from its length.

    This is protection against degenerate generation, not a duration estimate:
    the model stops at its own EOS in normal operation (verified token-identical
    to upstream). Sized for the slowest preset voice so it can never cut real
    speech; one conservative rate covers Latin and CJK alike.
    """
    seconds = max(1.0, len(str(text)) / _MIN_CHARS_PER_SECOND) * 1.5
    tokens = int(seconds * _CODES_PER_SECOND) + 32
    return max(int(floor), min(int(ceiling), tokens))


@dataclass
class Qwen3TTSSettings:
    """Sampling knobs; quality presets map onto these (both samplers together)."""

    do_sample: bool = True
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 1.0
    repetition_penalty: float = 1.05
    subtalker_temperature: float = 0.9
    subtalker_top_k: int = 50
    subtalker_top_p: float = 1.0
    quality_preset: str = "standard"

    def apply_quality_preset(self, preset: str) -> None:
        from ..quality_preset import normalize_quality_preset

        p = normalize_quality_preset(str(preset))
        self.quality_preset = p
        # Lower temperature/top_k = steadier, faster-converging output; "high"
        # keeps upstream's defaults, which their card tunes for expressiveness.
        if p == "low":
            self.temperature = 0.7
            self.top_k = 20
            self.subtalker_temperature = 0.7
            self.subtalker_top_k = 20
        elif p == "standard":
            self.temperature = 0.8
            self.top_k = 40
            self.subtalker_temperature = 0.8
            self.subtalker_top_k = 40
        else:  # high
            self.temperature = 0.9
            self.top_k = 50
            self.subtalker_temperature = 0.9
            self.subtalker_top_k = 50

    def generate_kwargs(self) -> dict:
        return dict(
            do_sample=self.do_sample,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            subtalker_temperature=self.subtalker_temperature,
            subtalker_top_k=self.subtalker_top_k,
            subtalker_top_p=self.subtalker_top_p,
        )


class Qwen3TTSRuntime:
    """Offline-first loader + synthesis entry points for one Qwen3-TTS snapshot."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        revision: Optional[str] = None,
        device: str = "auto",
        dtype_name: Optional[str] = None,
        allow_downloads: bool = True,
        debug: bool = False,
        settings: Optional[Qwen3TTSSettings] = None,
    ) -> None:
        self.model_id = str(model_id or DEFAULT_MODEL_ID)
        self.revision = revision
        self.allow_downloads = bool(allow_downloads)
        self.debug = bool(debug)
        self.settings = settings or Qwen3TTSSettings()

        self._device_pref = str(device or "auto")
        # The published checkpoints are BF16; fp16 overflows the codec's RVQ
        # epsilon division and the talker's logits. bf16 on accelerators,
        # fp32 on CPU, unless the caller pins something else. (ADR 0005: the
        # shared helpers resolve the device; this sets the *default* dtype.)
        self._dtype_pref = dtype_name

        self._model = None  # orchestration.Qwen3TTSModel
        self._resolved_device: Optional[str] = None
        self._resolved_dtype: Optional[str] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ loading

    def snapshot_dir(self, *, allow_downloads: Optional[bool] = None) -> str:
        """Resolve the local snapshot directory, offline-first.

        A local directory path is accepted as-is. Otherwise the Hugging Face
        cache is consulted; downloads happen only when downloads are allowed
        AND the caller is loading — introspection paths (``peek_config``,
        ``speaker_names``) always pass ``allow_downloads=False``, because
        listing voices must never fetch 2.5 GB as a side effect.
        """
        candidate = os.path.expanduser(self.model_id)
        if os.path.isdir(candidate):
            return candidate

        effective = self.allow_downloads if allow_downloads is None else bool(allow_downloads)

        try:
            from huggingface_hub import snapshot_download
        except Exception as e:  # pragma: no cover
            raise RuntimeError("huggingface_hub is required for Qwen3-TTS model management") from e

        try:
            return str(
                snapshot_download(
                    repo_id=self.model_id,
                    revision=self.revision,
                    local_files_only=not effective,
                )
            )
        except Exception as e:
            if not effective:
                raise RuntimeError(
                    "Qwen3-TTS weights are not available locally and downloads are disabled.\n"
                    "Fix options:\n"
                    "  - Enable downloads: VoiceManager(..., allow_downloads=True)\n"
                    "  - Or prefetch explicitly: abstractvoice-prefetch --qwen3-tts\n"
                    f"Model: {self.model_id}"
                ) from e
            raise

    def _resolve_runtime(self):
        default_dtype = self._dtype_pref
        if not default_dtype:
            probe = resolve_torch_runtime(
                device=self._device_pref,
                dtype_name=None,
                allow_cpu_fallback=self._device_pref == "auto",
            )
            default_dtype = "float32" if str(probe.resolved_device) == "cpu" else "bfloat16"
        runtime = resolve_torch_runtime(
            device=self._device_pref,
            dtype_name=default_dtype,
            allow_cpu_fallback=self._device_pref == "auto",
        )
        self._resolved_device = str(runtime.resolved_device)
        self._resolved_dtype = str(runtime.resolved_dtype_name)
        return runtime

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return

            import torch

            from .orchestration import Qwen3TTSModel

            local_dir = self.snapshot_dir()
            runtime = self._resolve_runtime()

            model = Qwen3TTSModel.from_pretrained(local_dir, dtype=runtime.torch_dtype)
            model.model.to(runtime.resolved_device)
            model.model.eval()
            # The codec pinned fp32 at load (see codec.py); move it to the device.
            codec = model.model.speech_tokenizer
            codec.model = codec.model.to(device=runtime.resolved_device)
            codec.device = model.model.device
            model.device = model.model.device
            self._model = model

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _model_or_raise(self):
        """The loaded model, read under the lock the caller already holds.

        A concurrent ``unload()`` between ``_ensure_loaded`` and the lock leaves
        ``_model`` None; that is a caller-visible lifecycle conflict, not a crash.
        """
        model = self._model
        if model is None:
            raise RuntimeError("Qwen3-TTS runtime was unloaded while a synthesis was starting; retry.")
        return model

    def unload(self) -> None:
        with self._lock:
            self._model = None
        try:
            import gc

            gc.collect()
            import torch

            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # ----------------------------------------------------------- introspection

    def model_type(self) -> str:
        """"custom_voice" | "base" | "voice_design", from the loaded model or its config."""
        if self._model is not None:
            return str(self._model.model.tts_model_type)
        return str(self.peek_config().get("tts_model_type") or "")

    def peek_config(self) -> dict:
        """Read config.json from the LOCAL snapshot without loading weights.

        Never downloads: introspection is a discovery operation.
        """
        import json

        from ..local_models import hf_cached_snapshot_dir

        snapshot = hf_cached_snapshot_dir(self.model_id)
        if snapshot is None:
            # Fall back to the resolver for local checkpoint dirs and clearer errors.
            snapshot = self.snapshot_dir(allow_downloads=False)
        path = os.path.join(str(snapshot), "config.json")
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def speaker_names(self) -> list[str]:
        """Preset speaker names, weight-free (from config's talker spk_id).

        Loaded and peek paths normalize identically (sorted, lowercase): the
        listing must not change shape because weights happen to be resident.
        """
        if self._model is not None:
            names = self._model.get_supported_speakers() or []
        else:
            cfg = self.peek_config()
            talker = cfg.get("talker_config") or {}
            names = list(talker.get("spk_id") or {})
        return sorted(str(name).lower() for name in names)

    def language_names(self) -> list[str]:
        """Supported language names; same loaded/peek coherence as speakers.

        The peek path replicates the loaded model's own rule (``["auto"]`` plus
        the non-dialect ``codec_language_id`` keys) rather than dumping raw
        config keys: the raw keys include dialect ids the model itself REJECTS
        at synthesis time, and omit ``auto`` -- a listing must not change set
        membership depending on whether weights happen to be resident.
        """
        if self._model is not None:
            names = list(self._model.get_supported_languages() or [])
        else:
            cfg = self.peek_config()
            talker = cfg.get("talker_config") or {}
            names = ["auto"] + [
                name for name in (talker.get("codec_language_id") or {}) if "dialect" not in str(name)
            ]
        return sorted(str(name).lower() for name in names)

    def runtime_info(self) -> dict:
        return {
            "model_id": self.model_id,
            "loaded": self.is_loaded,
            "device": self._resolved_device,
            "dtype": self._resolved_dtype,
            "codec_dtype": "float32",
            "model_type": self.model_type() if self.is_loaded else None,
            "quality_preset": self.settings.quality_preset,
        }

    # -------------------------------------------------------------- synthesis

    def synthesize_custom_voice(
        self,
        text: str,
        *,
        speaker: str,
        language: Optional[str] = None,
        instruct: Optional[str] = None,
        **overrides: Any,
    ) -> tuple[np.ndarray, int]:
        """One utterance with a preset speaker (CustomVoice models)."""
        self._ensure_loaded()
        kwargs = self.settings.generate_kwargs()
        kwargs["max_new_tokens"] = estimate_max_new_tokens(text)
        kwargs.update(overrides)
        with self._lock:
            wavs, sr = self._model_or_raise().generate_custom_voice(
                text=str(text),
                speaker=str(speaker),
                language=language or "Auto",
                instruct=instruct or None,
                **kwargs,
            )
        return np.asarray(wavs[0], dtype=np.float32), int(sr)

    def synthesize_voice_design(
        self,
        text: str,
        *,
        instruct: str,
        language: Optional[str] = None,
        **overrides: Any,
    ) -> tuple[np.ndarray, int]:
        """One utterance with a designed voice (VoiceDesign models).

        ``instruct`` is required: an empty description yields an arbitrary voice,
        which is a silent quality cliff rather than an error (ADR 0007 demands
        explicit degradation, so we refuse instead).
        """
        if not str(instruct or "").strip():
            raise ValueError(
                "Qwen3-TTS VoiceDesign requires a non-empty voice description "
                "(the `instructions` selector). Without one the voice is arbitrary."
            )
        self._ensure_loaded()
        kwargs = self.settings.generate_kwargs()
        kwargs["max_new_tokens"] = estimate_max_new_tokens(text)
        kwargs.update(overrides)
        with self._lock:
            wavs, sr = self._model_or_raise().generate_voice_design(
                text=str(text),
                instruct=str(instruct),
                language=language or "Auto",
                **kwargs,
            )
        return np.asarray(wavs[0], dtype=np.float32), int(sr)

    def build_clone_prompt(
        self,
        *,
        ref_audio: Any,
        ref_text: Optional[str],
        x_vector_only: bool,
    ):
        """Voice-clone prompt items from reference audio (Base models).

        The MODE IS THE CALLER'S DECISION (ADR 0003): ICL needs ``ref_text``,
        x-vector-only ignores it. This method never infers one from the other's
        absence — the cloning manager owns transcripts and their fallbacks.
        """
        self._ensure_loaded()
        with self._lock:
            return self._model_or_raise().create_voice_clone_prompt(
                ref_audio=ref_audio,
                ref_text=ref_text,
                x_vector_only_mode=bool(x_vector_only),
            )

    def synthesize_clone(
        self,
        text: str,
        *,
        clone_prompt: Any,
        language: Optional[str] = None,
        **overrides: Any,
    ) -> tuple[np.ndarray, int]:
        """One utterance in a cloned voice, from a prebuilt prompt (Base models)."""
        self._ensure_loaded()
        kwargs = self.settings.generate_kwargs()
        kwargs["max_new_tokens"] = estimate_max_new_tokens(text)
        kwargs.update(overrides)
        with self._lock:
            wavs, sr = self._model_or_raise().generate_voice_clone(
                text=str(text),
                language=language or "Auto",
                voice_clone_prompt=clone_prompt,
                **kwargs,
            )
        return np.asarray(wavs[0], dtype=np.float32), int(sr)


def prefetch_qwen3_tts(*, model_id: str = DEFAULT_MODEL_ID, allow_downloads: bool = True) -> str:
    """Explicitly download a Qwen3-TTS snapshot (never called implicitly)."""
    if not allow_downloads:
        raise RuntimeError("prefetch_qwen3_tts requires allow_downloads=True")
    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        raise RuntimeError(
            "Qwen3-TTS prefetch requires optional dependencies.\n"
            "Install with:\n"
            '  pip install "abstractvoice[qwen3-tts]"'
        ) from e
    return str(snapshot_download(str(model_id)))
