"""OmniVoice TTS adapter (optional).

Implements the AbstractVoice `TTSAdapter` contract by delegating to the
`omnivoice` Python package (Apache-2.0).

This adapter supports:
- omnilingual TTS via `language` (OmniVoice upstream supports 600+ languages)
- voice design via `instruct` (speaker attributes)
- native speed control via OmniVoice's `speed` parameter
"""

from __future__ import annotations

from dataclasses import asdict
import io
import importlib.util
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import soundfile as sf

from .base import TTSAdapter
from ..omnivoice.runtime import OmniVoiceRuntime, OmniVoiceSettings
from ..voice_profiles import VoiceProfile, find_voice_profile, get_builtin_voice_profiles


class OmniVoiceTTSAdapter(TTSAdapter):
    engine_id = "omnivoice"

    def __init__(
        self,
        *,
        language: str = "en",
        allow_downloads: bool = True,
        auto_load: bool = True,
        debug_mode: bool = False,
        model_id: str | None = None,
        revision: str | None = None,
        device: str = "auto",
    ):
        self.debug_mode = bool(debug_mode)
        self._language = str(language or "en").strip().lower() or "en"
        self._allow_downloads = bool(allow_downloads)

        self._runtime = OmniVoiceRuntime(
            model_id=model_id,
            revision=revision,
            device=str(device or "auto"),
            allow_downloads=bool(allow_downloads),
            debug=bool(debug_mode),
        )

        # Voice design instruction string (optional, adapter-level state).
        self._instruct: str | None = None
        # Optional fixed duration override (seconds). When set, OmniVoice ignores `speed`.
        self._duration_s: float | None = None
        # Best-effort active profile marker (set via `set_profile(...)`).
        self._active_profile_id: str | None = None

        # Engine-agnostic quality preset mapping.
        self._quality_preset = "balanced"
        self._settings = OmniVoiceSettings()
        self.set_quality_preset("balanced")

        if bool(auto_load):
            # Eagerly surface missing weights/deps for explicit engine selection.
            _ = self._runtime.get_model()

        # If built-in profiles exist, treat "default" as the baseline.
        # This is best-effort and must never fail adapter construction.
        try:
            self.set_profile("default")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Optional knobs (non-breaking)
    # ------------------------------------------------------------------

    def set_voice_design_instruct(self, instruct: str | None) -> bool:
        s = str(instruct).strip() if instruct is not None else ""
        self._instruct = s if s else None
        return True

    def set_fixed_duration_seconds(self, seconds: float | None) -> bool:
        if seconds is None:
            self._duration_s = None
            return True
        s = float(seconds)
        if s <= 0:
            raise ValueError("duration must be > 0 seconds (or omit/clear it)")
        self._duration_s = float(s)
        return True

    def get_params(self) -> Dict[str, Any]:
        """Return current OmniVoice generation parameters (best-effort)."""
        try:
            settings = asdict(getattr(self, "_settings", OmniVoiceSettings()))
        except Exception:
            settings = {}
        settings.update(
            {
                "quality_preset": str(getattr(self, "_quality_preset", "balanced") or "balanced"),
                "language": str(getattr(self, "_language", "en") or "en"),
                "instruct": (getattr(self, "_instruct", None) or None),
                "duration": (getattr(self, "_duration_s", None) or None),
            }
        )
        return dict(settings)

    # ------------------------------------------------------------------
    # Voice profile interface (engine-agnostic presets)
    # ------------------------------------------------------------------

    def get_profiles(self) -> list[VoiceProfile]:
        return list(get_builtin_voice_profiles(str(getattr(self, "engine_id", "omnivoice") or "omnivoice")))

    def set_profile(self, profile_id: str) -> bool:
        pid = str(profile_id or "").strip()
        if not pid:
            raise ValueError("profile_id must be a non-empty string")

        profiles = self.get_profiles()
        p = find_voice_profile(profiles, pid)
        if p is None:
            supported = ", ".join([pp.profile_id for pp in profiles]) if profiles else "(none)"
            raise ValueError(f"Unknown profile_id '{pid}'. Supported: {supported}")

        params = dict(getattr(p, "params", {}) or {})

        # Apply in a stable order for readability and predictability.
        if "quality_preset" in params:
            try:
                self.set_quality_preset(str(params.get("quality_preset") or "balanced"))
            except Exception:
                # Best-effort only; fall back to current preset.
                pass
        if "language" in params:
            try:
                self.set_language(str(params.get("language") or "").strip())
            except Exception:
                pass

        # Apply remaining engine params via the adapter's validated setter.
        for k, v in list(params.items()):
            kk = str(k or "").strip()
            if not kk or kk in ("quality_preset", "language"):
                continue
            try:
                self.set_param(kk, v)
            except Exception:
                # Profiles are best-effort; ignore unknown/invalid keys.
                continue

        self._active_profile_id = str(p.profile_id)
        return True

    def get_active_profile(self) -> VoiceProfile | None:
        pid = str(getattr(self, "_active_profile_id", None) or "").strip()
        profiles = self.get_profiles()
        if pid:
            p = find_voice_profile(profiles, pid)
            return p
        # If never explicitly set, treat "default" (when present) as the baseline.
        return find_voice_profile(profiles, "default")

    def _coerce_bool(self, value: Any) -> bool:
        if isinstance(value, bool):
            return bool(value)
        s = str(value).strip().lower()
        if s in ("1", "true", "yes", "y", "on"):
            return True
        if s in ("0", "false", "no", "n", "off"):
            return False
        raise ValueError("expected a boolean (on|off / true|false / 1|0)")

    def set_param(self, key: str, value: Any) -> bool:
        """Set a single OmniVoice parameter by name (best-effort).

        Supported keys:
        - Voice design: instruct
        - Reproducibility: seed (int; same seed + same params => same voice sampling)
        - Duration/speed: duration (seconds; overrides speed when set)
        - Generation: num_step, guidance_scale, t_shift, position_temperature,
          class_temperature, layer_penalty_factor, denoise, preprocess_prompt,
          postprocess_output, audio_chunk_duration, audio_chunk_threshold
        """
        k = str(key or "").strip().lower()
        if not k:
            raise ValueError("key must be a non-empty string")

        # Aliases (short names)
        aliases = {
            "steps": "num_step",
            "num_steps": "num_step",
            "step": "num_step",
            "guidance": "guidance_scale",
            "cfg": "guidance_scale",
            "pos_temp": "position_temperature",
            "position_temp": "position_temperature",
            "class_temp": "class_temperature",
            "layer_penalty": "layer_penalty_factor",
            "penalty": "layer_penalty_factor",
            "chunk_duration": "audio_chunk_duration",
            "chunk_threshold": "audio_chunk_threshold",
            "preprocess": "preprocess_prompt",
            "postprocess": "postprocess_output",
            "random_seed": "seed",
        }
        k = aliases.get(k, k)

        if k in ("instruct", "voice_design", "design"):
            return bool(self.set_voice_design_instruct(None if value is None else str(value)))
        if k in ("clear_instruct", "no_instruct"):
            return bool(self.set_voice_design_instruct(None))
        if k in ("duration", "duration_s", "seconds"):
            if value is None:
                return bool(self.set_fixed_duration_seconds(None))
            s = str(value).strip().lower()
            if s in ("", "none", "null", "off", "0"):
                return bool(self.set_fixed_duration_seconds(None))
            return bool(self.set_fixed_duration_seconds(float(value)))

        st = getattr(self, "_settings", None)
        if st is None:
            st = OmniVoiceSettings()
            self._settings = st

        if k in ("seed",):
            if value is None:
                st.seed = None
                return True
            s = str(value).strip().lower()
            if s in ("", "none", "null", "off"):
                st.seed = None
                return True
            try:
                st.seed = int(value)
            except Exception:
                st.seed = int(float(value))
            return True

        # Type coercion + basic validation
        if k == "num_step":
            n = int(value)
            if n <= 0:
                raise ValueError("num_step must be > 0")
            st.num_step = int(n)
            return True
        if k in ("guidance_scale", "t_shift", "position_temperature", "class_temperature", "layer_penalty_factor"):
            x = float(value)
            if x < 0:
                raise ValueError(f"{k} must be >= 0")
            setattr(st, k, float(x))
            return True
        if k in ("audio_chunk_duration", "audio_chunk_threshold"):
            x = float(value)
            if x <= 0:
                raise ValueError(f"{k} must be > 0")
            setattr(st, k, float(x))
            return True
        if k in ("denoise", "preprocess_prompt", "postprocess_output"):
            b = self._coerce_bool(value)
            setattr(st, k, bool(b))
            return True

        raise ValueError(f"Unknown OmniVoice param: {key}")

    # ------------------------------------------------------------------
    # TTSAdapter interface
    # ------------------------------------------------------------------

    def synthesize(self, text: str) -> np.ndarray:
        audio, _sr = self._runtime.generate_audio(
            text=str(text),
            language=str(self._language),
            instruct=self._instruct,
            duration=getattr(self, "_duration_s", None),
            speed=None,
            settings=getattr(self, "_settings", OmniVoiceSettings()),
        )
        return np.asarray(audio, dtype=np.float32).reshape(-1)

    def synthesize_with_speed(self, text: str, speed: float) -> np.ndarray:
        sp = float(speed) if speed is not None else 1.0
        audio, _sr = self._runtime.generate_audio(
            text=str(text),
            language=str(self._language),
            instruct=self._instruct,
            duration=getattr(self, "_duration_s", None),
            speed=sp,
            settings=getattr(self, "_settings", OmniVoiceSettings()),
        )
        return np.asarray(audio, dtype=np.float32).reshape(-1)

    def synthesize_to_bytes(self, text: str, format: str = "wav") -> bytes:
        if str(format or "wav").strip().lower() != "wav":
            raise ValueError("OmniVoice adapter currently only supports WAV output.")
        audio = self.synthesize(str(text))
        buf = io.BytesIO()
        sf.write(buf, np.asarray(audio, dtype=np.float32).reshape(-1), self.get_sample_rate(), format="WAV", subtype="PCM_16")
        return buf.getvalue()

    def synthesize_to_file(self, text: str, output_path: str, format: Optional[str] = None) -> str:
        out = Path(output_path)
        fmt = str(format or out.suffix.lstrip(".") or "wav").strip().lower()
        if fmt != "wav":
            raise ValueError("OmniVoice adapter currently only supports WAV output.")
        data = self.synthesize_to_bytes(str(text), format="wav")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        return str(out)

    def set_language(self, language: str) -> bool:
        # OmniVoice supports many languages; treat this as a soft setting.
        new_lang = str(language or "").strip().lower()
        if new_lang:
            self._language = new_lang
        return True

    def get_supported_languages(self) -> list[str]:
        # OmniVoice ships a large language id mapping (~600). Return it when the
        # package is installed; otherwise, fall back to common ISO-639-1 codes.
        try:
            from omnivoice.utils.lang_map import LANG_IDS  # type: ignore

            return sorted([str(x) for x in LANG_IDS])
        except Exception:
            return ["en", "fr", "de", "es", "ru", "zh", "ja", "ko"]

    def get_sample_rate(self) -> int:
        return int(self._runtime.get_sample_rate() or 24000)

    def is_available(self) -> bool:
        # Avoid loading multi-GB weights in a capability probe.
        try:
            if importlib.util.find_spec("torch") is None:
                return False
            if importlib.util.find_spec("torchaudio") is None:
                return False
            if importlib.util.find_spec("transformers") is None:
                return False
            if importlib.util.find_spec("omnivoice") is None:
                return False
        except Exception:
            return False
        return True

    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        info.update(
            {
                "engine": "OmniVoice",
                "engine_id": "omnivoice",
                "quality_preset": str(getattr(self, "_quality_preset", "balanced") or "balanced"),
                "language": str(getattr(self, "_language", "en") or "en"),
            }
        )
        try:
            info["runtime"] = dict(self._runtime.runtime_info())
        except Exception:
            pass
        return info

    def set_quality_preset(self, preset: str) -> bool:
        p = str(preset or "").strip().lower()
        if p not in ("fast", "balanced", "high"):
            raise ValueError("preset must be one of: fast|balanced|high")
        self._quality_preset = p

        # Map engine-agnostic presets to OmniVoice knobs.
        # Keep guidance_scale stable initially; steps are the main speed/quality driver.
        if p == "fast":
            self._settings.num_step = 16
            self._settings.guidance_scale = 2.0
        elif p == "balanced":
            self._settings.num_step = 32
            self._settings.guidance_scale = 2.0
        else:
            self._settings.num_step = 48
            self._settings.guidance_scale = 2.0
        return True

    def get_quality_preset(self) -> str | None:
        return str(getattr(self, "_quality_preset", None) or "balanced")

