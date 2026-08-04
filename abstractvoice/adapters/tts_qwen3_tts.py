"""Qwen3-TTS adapter (12Hz family: CustomVoice / VoiceDesign; cloning lives in
``abstractvoice/cloning/engine_qwen3_tts.py`` on the Base models).

Optional; requires:
  pip install "abstractvoice[qwen3-tts]"
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ..voice_profiles import VoiceProfile
from .base import TTSAdapter

# Our language codes -> the language names the checkpoints validate against.
# "auto" lets the model detect; the checkpoint list is read from its config.
_LANGUAGE_NAMES = {
    "en": "English",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "ru": "Russian",
    "pt": "Portuguese",
    "es": "Spanish",
    "it": "Italian",
}


def _qwen_language(code_or_name: Optional[str]) -> str:
    text = str(code_or_name or "").strip()
    if not text or text.lower() in {"auto", "default"}:
        return "Auto"
    return _LANGUAGE_NAMES.get(text.lower(), text.capitalize())


class Qwen3TTSAdapter(TTSAdapter):
    """TTS adapter over :class:`abstractvoice.qwen3_tts.runtime.Qwen3TTSRuntime`."""

    engine_id = "qwen3-tts"

    def __init__(
        self,
        language: str = "en",
        *,
        allow_downloads: bool = True,
        auto_load: bool = False,
        debug_mode: bool = False,
        model_id: str | None = None,
        revision: str | None = None,
        device: str = "auto",
        runtime: Any | None = None,
    ):
        self._language = str(language or "en").strip().lower()
        self._debug = bool(debug_mode)
        self._sample_rate = 24000
        self._speaker: Optional[str] = None
        self._instructions: Optional[str] = None

        if runtime is None:
            from ..qwen3_tts.runtime import DEFAULT_MODEL_ID, Qwen3TTSRuntime

            runtime = Qwen3TTSRuntime(
                model_id=model_id or DEFAULT_MODEL_ID,
                revision=revision,
                device=device,
                allow_downloads=bool(allow_downloads),
                debug=bool(debug_mode),
            )
        self._runtime = runtime
        # The VoiceManager mixin resolves per-model capability entries (e.g. the
        # 0.6B checkpoint ignoring `instructions`) from `adapter.model_id`.
        self.model_id = str(getattr(runtime, "model_id", "") or "")

        if bool(auto_load):
            # Surface load failures early when the engine was selected explicitly.
            self._runtime._ensure_loaded()

    # ------------------------------------------------------------- introspection

    def is_available(self) -> bool:
        # Capability probe: importable runtimes only, never multi-GB loads.
        try:
            import importlib.util

            return (
                importlib.util.find_spec("torch") is not None
                and importlib.util.find_spec("transformers") is not None
            )
        except Exception:
            return False

    def is_engine_loaded(self) -> bool:
        """Residency probe used by preload/unload bookkeeping."""
        return bool(getattr(self._runtime, "is_loaded", False))

    def unload(self) -> bool:
        """Release model memory; the generic unload path in the mixin calls this."""
        try:
            self._runtime.unload()
            return True
        except Exception:
            return False

    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        info.update(
            {
                "engine": "Qwen3-TTS (12Hz)",
                "engine_id": self.engine_id,
                "sample_rate": self.get_sample_rate(),
                "model_id": getattr(self._runtime, "model_id", None),
                "quality_preset": self.get_quality_preset(),
            }
        )
        try:
            info["runtime"] = dict(self._runtime.runtime_info())
        except Exception:
            pass
        return info

    def get_sample_rate(self) -> int:
        return int(self._sample_rate or 24000)

    def get_max_chars(self) -> int:
        # Chunk size bounds time-to-first-audio (ADR 0004). The slowest preset
        # renders ~4.8 chars/s, so 200 chars is ~40s of audio worst case and
        # ~13s for typical voices -- 400 meant over a minute per chunk.
        return 200

    # ----------------------------------------------------------------- language

    def set_language(self, language: str) -> bool:
        new_lang = str(language or "").strip().lower()
        if new_lang:
            self._language = new_lang
        return True

    def get_supported_languages(self) -> list[str]:
        try:
            names = {str(n).lower() for n in self._runtime.language_names()}
            codes = [code for code, name in _LANGUAGE_NAMES.items() if name.lower() in names]
            if codes:
                return codes
        except Exception:
            pass
        return list(_LANGUAGE_NAMES.keys())

    # ----------------------------------------------------------------- profiles

    def _speaker_names(self) -> list[str]:
        try:
            return [str(s) for s in self._runtime.speaker_names()]
        except Exception:
            return []

    def get_profiles(self) -> list[VoiceProfile]:
        """Preset speakers, read weight-free from the local snapshot's config.

        Empty for Base/VoiceDesign checkpoints (they have no preset speakers)
        and when the snapshot is not downloaded yet — discovery never fetches.
        """
        profiles: list[VoiceProfile] = []
        for name in self._speaker_names():
            profiles.append(
                VoiceProfile(
                    engine_id=self.engine_id,
                    profile_id=name,
                    label=name.replace("_", " ").title(),
                    description=f"Qwen3-TTS preset speaker {name}",
                    params={
                        "provider": self.engine_id,
                        "voice": name,
                        "model": str(getattr(self._runtime, "model_id", "") or ""),
                    },
                    tags={
                        "provider": self.engine_id,
                        "engine_id": self.engine_id,
                        "kind": "profile",
                        "cached": "true",
                    },
                )
            )
        return profiles

    def set_profile(self, profile_id: str) -> bool:
        name = str(profile_id or "").strip().lower()
        if not name:
            return False
        speakers = {s.lower() for s in self._speaker_names()}
        if speakers and name not in speakers:
            return False
        self._speaker = name
        return True

    def get_active_profile(self) -> VoiceProfile | None:
        for profile in self.get_profiles():
            if str(profile.profile_id).lower() == str(self._active_speaker() or "").lower():
                return profile
        return None

    def get_default_profile_id(self, language: str | None = None) -> str | None:
        _ = language
        speakers = self._speaker_names()
        return speakers[0] if speakers else None

    def _active_speaker(self) -> Optional[str]:
        if self._speaker:
            return self._speaker
        speakers = self._speaker_names()
        return speakers[0] if speakers else None

    # ----------------------------------------------------------- quality preset

    def set_quality_preset(self, preset: str) -> bool:
        try:
            self._runtime.settings.apply_quality_preset(str(preset))
            return True
        except Exception:
            return False

    def get_quality_preset(self) -> Optional[str]:
        try:
            return str(self._runtime.settings.quality_preset)
        except Exception:
            return None

    # -------------------------------------------------------------- instructions

    def set_instructions(self, instructions: Optional[str]) -> bool:
        """Session-level style instructions (native on 1.7B checkpoints)."""
        text = str(instructions or "").strip()
        self._instructions = text or None
        return True

    # ---------------------------------------------------------------- synthesis

    def synthesize(self, text: str) -> np.ndarray:
        return self._synthesize_array(text, speaker=self._active_speaker(), instructions=self._instructions)

    def _synthesize_array(
        self,
        text: str,
        *,
        speaker: Optional[str],
        instructions: Optional[str],
    ) -> np.ndarray:
        model_type = self._model_type_or_raise()
        language = _qwen_language(self._language)

        if model_type == "custom_voice":
            if not speaker:
                raise RuntimeError(
                    "Qwen3-TTS CustomVoice needs a speaker profile; none are visible. "
                    "Prefetch the model (abstractvoice-prefetch --qwen3-tts) so its "
                    "speaker list can be read."
                )
            audio, sr = self._runtime.synthesize_custom_voice(
                str(text), speaker=str(speaker), language=language, instruct=instructions
            )
        elif model_type == "voice_design":
            # Instructions are REQUIRED here; the runtime raises the explicit
            # error (ADR 0007: refuse rather than silently produce an arbitrary voice).
            audio, sr = self._runtime.synthesize_voice_design(
                str(text), instruct=str(instructions or ""), language=language
            )
        else:
            raise RuntimeError(
                f"Qwen3-TTS model type {model_type!r} is a cloning checkpoint; select it "
                "through voice cloning (cloning_engine='qwen3-tts'), not as base TTS."
            )
        self._sample_rate = int(sr)
        return np.asarray(audio, dtype=np.float32).reshape(-1)

    def _model_type_or_raise(self) -> str:
        model_type = str(self._runtime.model_type() or "").strip().lower()
        if not model_type:
            raise RuntimeError(
                "Could not determine the Qwen3-TTS model type. Prefetch the model first: "
                "abstractvoice-prefetch --qwen3-tts"
            )
        return model_type

    def synthesize_to_bytes(self, text: str, format: str = "wav") -> bytes:
        if str(format or "wav").strip().lower() != "wav":
            raise ValueError("Qwen3-TTS adapter currently supports WAV output only.")
        import soundfile as sf

        audio = self.synthesize(str(text))
        buf = io.BytesIO()
        sf.write(buf, audio, int(self.get_sample_rate()), format="WAV", subtype="PCM_16")
        return buf.getvalue()

    def synthesize_to_bytes_with_voice(
        self,
        text: str,
        *,
        format: str = "wav",
        voice: str | None = None,
        speed: Any = None,
        instructions: str | None = None,
        **_kwargs: Any,
    ) -> bytes:
        """Voice/instructions-aware synthesis used by the VoiceManager mixin."""
        _ = speed  # Speed is reported unsupported in the capability catalog.
        if str(format or "wav").strip().lower() != "wav":
            raise ValueError("Qwen3-TTS adapter currently supports WAV output only.")
        speaker = self._active_speaker()
        if voice and str(voice).strip():
            requested = str(voice).strip().lower()
            speakers = {s.lower() for s in self._speaker_names()}
            if speakers and requested not in speakers:
                raise ValueError(f"Unknown Qwen3-TTS speaker: {voice}")
            speaker = requested  # per-call override; session profile untouched
        effective_instructions = (
            str(instructions).strip() if instructions is not None and str(instructions).strip() else self._instructions
        )
        import soundfile as sf

        audio = self._synthesize_array(str(text), speaker=speaker, instructions=effective_instructions)
        buf = io.BytesIO()
        sf.write(buf, audio, int(self.get_sample_rate()), format="WAV", subtype="PCM_16")
        return buf.getvalue()

    def synthesize_to_file(self, text: str, output_path: str, format: Optional[str] = None) -> str:
        fmt = (format or Path(output_path).suffix.lstrip(".") or "wav").strip().lower()
        if fmt != "wav":
            raise ValueError("Qwen3-TTS adapter currently supports WAV output only.")
        data = self.synthesize_to_bytes(str(text), format="wav")
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        return str(out)


def cached_qwen3_tts_voice_profiles() -> list[VoiceProfile]:
    """Preset-speaker profiles for the Qwen3-TTS snapshots on this machine.

    Weight-free and download-free (mirrors ``cached_piper_voice_profiles``): the
    speaker list lives in each CustomVoice snapshot's config.json, so the light
    catalog can carry real voices without importing torch or loading a model.
    """
    import json

    profiles: list[VoiceProfile] = []
    try:
        from ..local_models import hf_cached_snapshot_dir
        from ..qwen3_tts import KNOWN_MODEL_IDS
    except Exception:
        return profiles

    for model_id in KNOWN_MODEL_IDS:
        snapshot = hf_cached_snapshot_dir(model_id)
        if snapshot is None:
            continue
        try:
            with open(snapshot / "config.json", "r", encoding="utf-8") as fh:
                cfg = json.load(fh)
        except Exception:
            continue
        if str(cfg.get("tts_model_type") or "") != "custom_voice":
            continue
        speakers = ((cfg.get("talker_config") or {}).get("spk_id") or {})
        for name in sorted(str(s).lower() for s in speakers):
            profiles.append(
                VoiceProfile(
                    engine_id="qwen3-tts",
                    profile_id=name,
                    label=name.replace("_", " ").title(),
                    description=f"Qwen3-TTS preset speaker {name}",
                    params={"provider": "qwen3-tts", "voice": name, "model": model_id},
                    tags={
                        "provider": "qwen3-tts",
                        "engine_id": "qwen3-tts",
                        "kind": "profile",
                        "cached": "true",
                    },
                )
            )
    return profiles
