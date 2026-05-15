"""Supertonic 3 TTS adapter.

This adapter uses AbstractVoice's internal ONNX runtime wrapper and does not
depend on the external ``supertonic`` Python SDK.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .base import TTSAdapter
from ..supertonic.runtime import (
    DEFAULT_REVISION,
    MODEL_ID,
    SUPERTONIC_LANGUAGES,
    SUPERTONIC_VOICE_STYLES,
    SupertonicRuntime,
    get_supertonic_cache_dir,
    is_supertonic_cached,
)
from ..voice_profiles import VoiceProfile


class SupertonicTTSAdapter(TTSAdapter):
    """Adapter for Supertone Supertonic 3 fixed-style ONNX TTS."""

    engine_id = "supertonic"

    def __init__(
        self,
        language: str = "en",
        *,
        allow_downloads: bool = True,
        auto_load: bool = True,
        debug_mode: bool = False,
        model_id: str | None = None,
        revision: str | None = None,
        cache_dir: str | None = None,
        runtime: SupertonicRuntime | None = None,
        voice_style: str = "M1",
        max_chars: int | None = None,
    ):
        self.debug_mode = bool(debug_mode)
        self._language = str(language or "en").strip().lower() or "en"
        if self._language not in SUPERTONIC_LANGUAGES:
            self._language = "na"
        self._allow_downloads = bool(allow_downloads)
        self._model_id = str(model_id or "supertonic-3").strip() or "supertonic-3"
        self._revision = str(revision or DEFAULT_REVISION).strip() or DEFAULT_REVISION
        self._cache_dir = str(get_supertonic_cache_dir(cache_dir))
        self._runtime = runtime
        self._active_style = self._normalize_style(voice_style)
        self._sample_rate = 24000
        self._quality_preset = "standard"
        self._steps = 8
        self._speed = 1.05
        self._max_chars = int(max_chars) if isinstance(max_chars, int) and int(max_chars) > 0 else None
        self._onnx_available = importlib.util.find_spec("onnxruntime") is not None

        # Keep construction/catalog discovery side-effect-light. If artifacts are
        # cached, eager load catches corrupt/missing files early; otherwise the
        # first synthesis call or explicit prefetch performs the download.
        if bool(auto_load) and is_supertonic_cached(self._cache_dir):
            try:
                self._get_runtime()
            except Exception:
                if self._allow_downloads:
                    raise

    @staticmethod
    def _normalize_style(style: str | None) -> str:
        value = str(style or "M1").strip().upper() or "M1"
        return value if value in SUPERTONIC_VOICE_STYLES else "M1"

    def _get_runtime(self) -> SupertonicRuntime:
        if self._runtime is None:
            if not self._onnx_available:
                raise RuntimeError(
                    "Supertonic requires ONNX Runtime.\n"
                    "Install with:\n"
                    '  pip install "abstractvoice[supertonic]"'
                )
            self._runtime = SupertonicRuntime(
                cache_dir=self._cache_dir,
                revision=self._revision,
                allow_downloads=bool(self._allow_downloads),
            )
        # Force load so missing cache produces an actionable error here.
        self._runtime._ensure_loaded()  # noqa: SLF001
        self._sample_rate = int(self._runtime.sample_rate)
        return self._runtime

    def ensure_model_downloaded(self) -> bool:
        rt = self._runtime or SupertonicRuntime(
            cache_dir=self._cache_dir,
            revision=self._revision,
            allow_downloads=True,
        )
        rt.ensure_downloaded()
        self._runtime = rt
        return True

    def get_max_chars(self) -> int:
        if self._max_chars is not None:
            return int(self._max_chars)
        return 120 if self._language in {"ko", "ja"} else 300

    def set_language(self, language: str) -> bool:
        lang = str(language or "").strip().lower()
        if not lang:
            return False
        if lang not in SUPERTONIC_LANGUAGES and lang != "na":
            return False
        self._language = lang
        return True

    def get_supported_languages(self) -> list[str]:
        return list(SUPERTONIC_LANGUAGES)

    def get_sample_rate(self) -> int:
        return int(self._sample_rate or 24000)

    def is_available(self) -> bool:
        return bool(self._onnx_available and is_supertonic_cached(self._cache_dir))

    def get_unavailable_reason(self) -> str | None:
        if not bool(self._onnx_available):
            return (
                "Supertonic requires ONNX Runtime.\n"
                "Install with:\n"
                '  pip install "abstractvoice[supertonic]"\n'
                '  pip install "abstractvoice[apple]"  # Apple profile\n'
                '  pip install "abstractvoice[gpu]"    # GPU profile\n'
                '  pip install "abstractvoice[all-apple]"  # Apple profile + web\n'
                '  pip install "abstractvoice[all-gpu]"    # GPU profile + web'
            )
        if not bool(is_supertonic_cached(self._cache_dir)):
            return (
                "Supertonic 3 artifacts are not available locally.\n"
                "Run: python -m abstractvoice download --supertonic\n"
                "Or in the REPL: /tts_download supertonic"
            )
        return None

    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        info.update(
            {
                "engine": "Supertonic 3",
                "engine_id": "supertonic",
                "model_id": MODEL_ID,
                "revision": self._revision,
                "cache_dir": self._cache_dir,
                "cached": bool(is_supertonic_cached(self._cache_dir)),
                "current_language": self._language,
                "active_profile": self._active_style,
                "quality_preset": self._quality_preset,
                "onnxruntime_available": bool(self._onnx_available),
            }
        )
        return info

    def set_quality_preset(self, preset: str) -> bool:
        from ..quality_preset import normalize_quality_preset

        p = normalize_quality_preset(str(preset))
        self._quality_preset = str(p)
        if p == "low":
            self._steps = 5
        elif p == "standard":
            self._steps = 8
        else:
            self._steps = 12
        return True

    def get_quality_preset(self) -> str | None:
        return str(self._quality_preset or "standard")

    def get_profiles(self) -> list[VoiceProfile]:
        cached = bool(is_supertonic_cached(self._cache_dir))
        profiles: list[VoiceProfile] = []
        for style in SUPERTONIC_VOICE_STYLES:
            gender = "female" if style.startswith("F") else "male"
            profiles.append(
                VoiceProfile(
                    engine_id="supertonic",
                    profile_id=style,
                    label=f"Supertonic {style}",
                    description=f"Supertonic 3 {gender} fixed voice style",
                    params={
                        "provider": "supertonic",
                        "model": "supertonic-3",
                        "voice": style,
                        "language": self._language,
                    },
                    tags={
                        "provider": "supertonic",
                        "engine_id": "supertonic",
                        "kind": "profile",
                        "gender": gender,
                        "cached": "true" if cached else "false",
                    },
                    provenance={
                        "source": "Supertone/supertonic-3",
                        "revision": self._revision,
                        "license": "openrail",
                    },
                )
            )
        return profiles

    def set_profile(self, profile_id: str) -> bool:
        requested = str(profile_id or "").strip().upper()
        if requested not in SUPERTONIC_VOICE_STYLES:
            return False
        self._active_style = requested
        return True

    def get_default_profile_id(self, language: str | None = None) -> str | None:
        _ = language
        return "M1"

    def get_active_profile(self) -> VoiceProfile | None:
        for profile in self.get_profiles():
            if profile.profile_id == self._active_style:
                return profile
        return None

    def _synthesize_with_speed(self, text: str, speed: float | None = None) -> np.ndarray:
        rt = self._get_runtime()
        audio = rt.synthesize(
            str(text),
            language=str(self._language),
            voice_style=str(self._active_style),
            total_steps=int(self._steps),
            speed=float(speed if speed is not None else self._speed),
            max_chars=int(self.get_max_chars()),
        )
        self._sample_rate = int(rt.sample_rate)
        return np.asarray(audio, dtype=np.float32).reshape(-1)

    def synthesize(self, text: str) -> np.ndarray:
        return self._synthesize_with_speed(text, self._speed)

    def synthesize_with_speed(self, text: str, speed: float) -> np.ndarray:
        return self._synthesize_with_speed(text, float(speed))

    def synthesize_to_audio_chunks(self, text: str):
        rt = self._get_runtime()
        for chunk, sr in rt.iter_audio_chunks(
            str(text),
            language=str(self._language),
            voice_style=str(self._active_style),
            total_steps=int(self._steps),
            speed=float(self._speed),
            max_chars=int(self.get_max_chars()),
        ):
            self._sample_rate = int(sr)
            mono = np.asarray(chunk, dtype=np.float32).reshape(-1)
            if mono.size:
                yield mono, int(sr)

    def synthesize_to_bytes(self, text: str, format: str = "wav") -> bytes:
        fmt = str(format or "wav").strip().lower() or "wav"
        if fmt != "wav":
            raise ValueError("Supertonic adapter currently supports WAV output only.")
        rt = self._get_runtime()
        audio = self._synthesize_with_speed(str(text), self._speed)
        return rt.synthesize_to_wav_bytes(audio)

    def synthesize_to_file(self, text: str, output_path: str, format: Optional[str] = None) -> str:
        fmt = (format or Path(output_path).suffix.lstrip(".") or "wav").strip().lower()
        if fmt != "wav":
            raise ValueError("Supertonic adapter currently supports WAV output only.")
        data = self.synthesize_to_bytes(str(text), format="wav")
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        return str(out)

    def synthesize_to_bytes_with_voice(self, text: str, *, format: str = "wav", voice: str | None = None, **_kwargs: Any) -> bytes:
        if voice and not self.set_profile(str(voice)):
            raise ValueError(f"Unknown Supertonic voice/profile: {voice}")
        return self.synthesize_to_bytes(text, format=format)

    def list_available_models(self, language: Optional[str] = None) -> Dict[str, Any]:
        cached = bool(is_supertonic_cached(self._cache_dir))
        group = str(language or self._language or "supertonic-3").strip().lower() or "supertonic-3"
        if group == "na":
            group = "supertonic-3"
        voices: Dict[str, Any] = {}
        for style in SUPERTONIC_VOICE_STYLES:
            voices[style] = {
                "name": f"Supertonic {style}",
                "quality": str(self._quality_preset),
                "size_mb": 386,
                "description": "Supertonic 3 fixed voice style",
                "requires_espeak": False,
                "cached": cached,
                "model_filename": "supertonic-3",
                "voice": style,
            }
        return {group: voices}
