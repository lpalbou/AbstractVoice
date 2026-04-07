"""OmniVoice TTS adapter (optional).

Implements the AbstractVoice `TTSAdapter` contract by delegating to the
`omnivoice` Python package (Apache-2.0).

This adapter supports:
- omnilingual TTS via `language` (OmniVoice upstream supports 600+ languages)
- voice design via `instruct` (speaker attributes)
- native speed control via OmniVoice's `speed` parameter
"""

from __future__ import annotations

import copy
from dataclasses import asdict
import io
import importlib.util
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import soundfile as sf

from .base import TTSAdapter
from ..omnivoice.runtime import OmniVoiceRuntime, OmniVoiceSettings
from ..omnivoice.prompt_cache import (
    analyze_prompt_audio_mono,
    get_omnivoice_prompt_cache_dir,
    load_cached_omnivoice_prompt,
    save_cached_omnivoice_prompt,
)
from ..voice_profiles import VoiceProfile, find_voice_profile, get_builtin_voice_profiles


class OmniVoiceTTSAdapter(TTSAdapter):
    engine_id = "omnivoice"
    _PROMPT_BUILD_ALGO_VERSION = 2

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
        # Persistent voice profile prompt (cached `voice_clone_prompt` tokens).
        # When enabled, synthesis uses `voice_clone_prompt` instead of re-sampling
        # a voice from `instruct` on every utterance.
        self._profile_prompt_enabled: bool = False
        self._voice_clone_prompt = None
        self._voice_clone_prompt_cache_dir: str | None = None

        # Engine-agnostic quality preset mapping.
        self._quality_preset = "standard"
        self._settings = OmniVoiceSettings()
        self.set_quality_preset("standard")

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
                "quality_preset": str(getattr(self, "_quality_preset", "standard") or "standard"),
                "language": str(getattr(self, "_language", "en") or "en"),
                "instruct": (getattr(self, "_instruct", None) or None),
                "duration": (getattr(self, "_duration_s", None) or None),
                "persistent_prompt": bool(getattr(self, "_profile_prompt_enabled", False)),
                "persistent_prompt_cached": bool(getattr(self, "_voice_clone_prompt", None) is not None),
                "persistent_prompt_cache_dir": (getattr(self, "_voice_clone_prompt_cache_dir", None) or None),
            }
        )
        return dict(settings)

    # ------------------------------------------------------------------
    # Voice profile interface (engine-agnostic presets)
    # ------------------------------------------------------------------

    def get_profiles(self) -> list[VoiceProfile]:
        return list(get_builtin_voice_profiles(str(getattr(self, "engine_id", "omnivoice") or "omnivoice")))

    def _extract_prompt_cache_config(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Extract (and remove) profile prompt-cache settings from params."""
        cfg: Dict[str, Any] = {}

        # Primary knob: `prompt_cache` can be a bool-like or a dict.
        if "prompt_cache" in params:
            raw = params.pop("prompt_cache", None)
            if isinstance(raw, dict):
                cfg.update(dict(raw))
            else:
                try:
                    cfg["enabled"] = self._coerce_bool(raw)
                except Exception:
                    cfg["enabled"] = bool(raw)

        # Alternate name (simple bool-like).
        if "persistent_prompt" in params and "enabled" not in cfg:
            raw = params.pop("persistent_prompt", None)
            try:
                cfg["enabled"] = self._coerce_bool(raw)
            except Exception:
                cfg["enabled"] = bool(raw)

        # Flat keys for convenience in JSON profile packs.
        for flat_key, cfg_key in (
            ("prompt_text", "text"),
            ("prompt_duration_s", "duration_s"),
            ("prompt_attempts", "attempts"),
            ("prompt_save_wav", "save_wav"),
            # Prompt-build overrides (independent from synthesis settings).
            ("prompt_position_temperature", "position_temperature"),
            ("prompt_class_temperature", "class_temperature"),
            ("prompt_num_step", "num_step"),
            ("prompt_guidance_scale", "guidance_scale"),
        ):
            if flat_key in params:
                cfg[cfg_key] = params.pop(flat_key)

        # Normalize defaults.
        enabled = bool(cfg.get("enabled") or False)
        try:
            attempts = int(cfg.get("attempts") or (4 if enabled else 1))
        except Exception:
            attempts = 4 if enabled else 1
        if attempts < 1:
            attempts = 1
        cfg["enabled"] = enabled
        cfg["attempts"] = int(attempts)

        cfg["text"] = str(cfg.get("text") or "").strip()

        try:
            dur = cfg.get("duration_s", None)
            if dur is None:
                cfg["duration_s"] = None
            else:
                d = float(dur)
                cfg["duration_s"] = None if d <= 0 else float(d)
        except Exception:
            cfg["duration_s"] = None

        try:
            cfg["save_wav"] = self._coerce_bool(cfg.get("save_wav", True))
        except Exception:
            cfg["save_wav"] = True

        # Prompt-build defaults tuned for intelligible voice-design prompts.
        # These are intentionally separate from profile synthesis parameters.
        try:
            pt = cfg.get("position_temperature", None)
            cfg["position_temperature"] = float(pt) if pt is not None else 1.0
        except Exception:
            cfg["position_temperature"] = 1.0
        try:
            ct = cfg.get("class_temperature", None)
            cfg["class_temperature"] = float(ct) if ct is not None else 0.5
        except Exception:
            cfg["class_temperature"] = 0.5
        try:
            ns = cfg.get("num_step", None)
            n = int(ns) if ns is not None else 48
            cfg["num_step"] = int(n) if int(n) > 0 else 48
        except Exception:
            cfg["num_step"] = 48
        try:
            gs = cfg.get("guidance_scale", None)
            g = float(gs) if gs is not None else 2.0
            cfg["guidance_scale"] = float(g) if float(g) >= 0 else 2.0
        except Exception:
            cfg["guidance_scale"] = 2.0

        return cfg

    def _prompt_spec(
        self,
        *,
        profile: VoiceProfile,
        prompt_text: str,
        prompt_duration_s: float | None,
        prompt_build: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        # Keep this JSON-serializable and stable across versions.
        try:
            st = asdict(getattr(self, "_settings", OmniVoiceSettings()))
        except Exception:
            st = {}
        return {
            "engine_id": "omnivoice",
            "profile_id": str(getattr(profile, "profile_id", "") or ""),
            "language": str(getattr(self, "_language", "en") or "en"),
            "prompt_text": str(prompt_text),
            "prompt_duration_s": float(prompt_duration_s) if prompt_duration_s is not None else None,
            "prompt_instruct": (getattr(self, "_instruct", None) or None),
            "prompt_seed": getattr(getattr(self, "_settings", None), "seed", None),
            "prompt_build_algo_version": int(getattr(self, "_PROMPT_BUILD_ALGO_VERSION", 2)),
            "prompt_build": dict(prompt_build or {}),
            "settings": dict(st),
            "model_id": str(getattr(getattr(self, "_runtime", None), "model_id", "") or ""),
            "revision": str(getattr(getattr(self, "_runtime", None), "revision", "") or "") or None,
        }

    def _cached_prompt_to_voice_clone_prompt(self, cached) -> Any:
        """Convert cached token payload to an object OmniVoice can consume."""
        import torch
        from dataclasses import dataclass

        toks = np.asarray(getattr(cached, "ref_audio_tokens", None))
        if toks.ndim == 1:
            toks = toks.reshape(1, -1)
        t = torch.from_numpy(toks.astype(np.int64, copy=False)).to(dtype=torch.long)

        @dataclass(frozen=True)
        class _VoiceClonePromptLite:
            ref_audio_tokens: Any
            ref_text: str
            ref_rms: float

        return _VoiceClonePromptLite(
            ref_audio_tokens=t,
            ref_text=str(getattr(cached, "ref_text", "") or ""),
            ref_rms=float(getattr(cached, "ref_rms", 0.0) or 0.0),
        )

    def _ensure_persistent_profile_prompt(self, profile: VoiceProfile, cfg: Dict[str, Any]) -> None:
        """Ensure `self._voice_clone_prompt` is populated for the active profile."""
        prompt_text = str(cfg.get("text") or "").strip()
        if not prompt_text:
            # Keep this short so voice-design prompts remain intelligible.
            prompt_text = "The quick brown fox jumps over the lazy dog."

        prompt_duration_s = cfg.get("duration_s", None)
        attempts = int(cfg.get("attempts") or 1)
        save_wav = bool(cfg.get("save_wav") if "save_wav" in cfg else True)
        prompt_build = {
            "position_temperature": float(cfg.get("position_temperature", 1.0)),
            "class_temperature": float(cfg.get("class_temperature", 0.5)),
            "num_step": int(cfg.get("num_step", 48)),
            "guidance_scale": float(cfg.get("guidance_scale", 2.0)),
        }

        runtime = getattr(self, "_runtime", None)
        if runtime is None:
            raise RuntimeError("OmniVoice runtime is not initialized")

        model_id = str(getattr(runtime, "model_id", "") or "")
        revision = getattr(runtime, "revision", None)
        lang = str(getattr(self, "_language", "en") or "en")
        cache_dir = get_omnivoice_prompt_cache_dir(
            model_id=model_id,
            revision=str(revision) if revision else None,
            language=lang,
            profile_id=str(getattr(profile, "profile_id", "") or ""),
        )

        spec = self._prompt_spec(
            profile=profile,
            prompt_text=prompt_text,
            prompt_duration_s=(float(prompt_duration_s) if prompt_duration_s is not None else None),
            prompt_build=dict(prompt_build),
        )

        cached = load_cached_omnivoice_prompt(cache_dir, expected_prompt_spec=spec)
        if cached is not None:
            self._voice_clone_prompt = self._cached_prompt_to_voice_clone_prompt(cached)
            self._voice_clone_prompt_cache_dir = str(cache_dir)
            self._profile_prompt_enabled = True
            return

        # Cache miss: build a prompt (one-time cost).
        model = runtime.get_model()

        base_seed = getattr(getattr(self, "_settings", None), "seed", None)
        candidates = []
        for i in range(int(attempts)):
            st = copy.deepcopy(getattr(self, "_settings", OmniVoiceSettings()))
            if base_seed is not None:
                try:
                    st.seed = int(base_seed) + int(i)
                except Exception:
                    st.seed = base_seed
            # Prompt-build overrides: tune voice-design sampling for intelligibility.
            try:
                st.position_temperature = float(prompt_build.get("position_temperature", 1.0))
            except Exception:
                st.position_temperature = 1.0
            try:
                st.class_temperature = float(prompt_build.get("class_temperature", 0.5))
            except Exception:
                st.class_temperature = 0.5
            try:
                st.num_step = int(prompt_build.get("num_step", 48))
            except Exception:
                st.num_step = 48
            try:
                st.guidance_scale = float(prompt_build.get("guidance_scale", 2.0))
            except Exception:
                st.guidance_scale = 2.0

            audio, sr = runtime.generate_audio(
                text=str(prompt_text),
                language=str(lang),
                instruct=getattr(self, "_instruct", None),
                voice_clone_prompt=None,
                duration=(float(prompt_duration_s) if prompt_duration_s is not None else None),
                speed=None,
                settings=st,
            )
            mono = np.asarray(audio, dtype=np.float32).reshape(-1)
            metrics = analyze_prompt_audio_mono(mono, int(sr))
            candidates.append((metrics, mono, int(sr)))

        # Choose the "least glitchy" candidate by lightweight heuristics.
        def _key(item):
            m = item[0] or {}
            return (float(m.get("hf_ratio_6k", 0.0)), float(m.get("p99_diff", 0.0)))

        candidates = [c for c in candidates if c[1].size > 0]
        if not candidates:
            raise RuntimeError("Failed to build a reference prompt for persistent profile")
        best_m, best_audio, best_sr = sorted(candidates, key=_key)[0]

        # IMPORTANT: keep AbstractVoice lightweight by passing (waveform, sr) directly.
        import torch

        waveform = torch.from_numpy(np.asarray(best_audio, dtype=np.float32)).unsqueeze(0)  # (1, T)
        prompt = model.create_voice_clone_prompt(
            ref_audio=(waveform, int(best_sr)),
            ref_text=str(prompt_text),
            preprocess_prompt=bool(getattr(getattr(self, "_settings", None), "preprocess_prompt", True)),
        )

        try:
            tokens = prompt.ref_audio_tokens.detach().cpu().numpy()
        except Exception:
            tokens = np.asarray(prompt.ref_audio_tokens)

        save_cached_omnivoice_prompt(
            cache_dir,
            ref_audio_tokens=np.asarray(tokens),
            ref_text=str(prompt_text),
            ref_rms=float(getattr(prompt, "ref_rms", 0.0) or 0.0),
            prompt_spec=spec,
            extra_meta={"prompt_metrics": dict(best_m) if isinstance(best_m, dict) else {}},
        )

        if save_wav:
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
                sf.write(
                    str(cache_dir / "prompt.wav"),
                    np.asarray(best_audio, dtype=np.float32).reshape(-1),
                    int(best_sr),
                    format="WAV",
                    subtype="PCM_16",
                )
            except Exception:
                pass

        cached2 = load_cached_omnivoice_prompt(cache_dir, expected_prompt_spec=spec)
        if cached2 is None:
            self._voice_clone_prompt = prompt
        else:
            self._voice_clone_prompt = self._cached_prompt_to_voice_clone_prompt(cached2)

        self._voice_clone_prompt_cache_dir = str(cache_dir)
        self._profile_prompt_enabled = True

    def set_profile(self, profile_id: str) -> bool:
        pid = str(profile_id or "").strip()
        if not pid:
            raise ValueError("profile_id must be a non-empty string")

        # Best-effort atomicity: restore previous adapter state on failure.
        prev = {
            "_language": getattr(self, "_language", "en"),
            "_quality_preset": getattr(self, "_quality_preset", "standard"),
            "_settings": copy.deepcopy(getattr(self, "_settings", OmniVoiceSettings())),
            "_instruct": getattr(self, "_instruct", None),
            "_duration_s": getattr(self, "_duration_s", None),
            "_active_profile_id": getattr(self, "_active_profile_id", None),
            "_profile_prompt_enabled": bool(getattr(self, "_profile_prompt_enabled", False)),
            "_voice_clone_prompt": getattr(self, "_voice_clone_prompt", None),
            "_voice_clone_prompt_cache_dir": getattr(self, "_voice_clone_prompt_cache_dir", None),
        }

        profiles = self.get_profiles()
        p = find_voice_profile(profiles, pid)
        if p is None:
            supported = ", ".join([pp.profile_id for pp in profiles]) if profiles else "(none)"
            raise ValueError(f"Unknown profile_id '{pid}'. Supported: {supported}")

        params = dict(getattr(p, "params", {}) or {})
        prompt_cfg = self._extract_prompt_cache_config(params)

        # Apply in a stable order for readability and predictability.
        if "quality_preset" in params:
            try:
                self.set_quality_preset(str(params.get("quality_preset") or "standard"))
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

        # Configure persistent prompt caching (if enabled by the profile).
        try:
            if bool(prompt_cfg.get("enabled") or False):
                self._ensure_persistent_profile_prompt(p, prompt_cfg)
            else:
                self._profile_prompt_enabled = False
                self._voice_clone_prompt = None
                self._voice_clone_prompt_cache_dir = None
        except Exception as e:
            # Restore previous state to keep `set_profile` atomic-ish.
            try:
                self._language = prev["_language"]
                self._quality_preset = prev["_quality_preset"]
                self._settings = prev["_settings"]
                self._instruct = prev["_instruct"]
                self._duration_s = prev["_duration_s"]
                self._active_profile_id = prev["_active_profile_id"]
                self._profile_prompt_enabled = prev["_profile_prompt_enabled"]
                self._voice_clone_prompt = prev["_voice_clone_prompt"]
                self._voice_clone_prompt_cache_dir = prev["_voice_clone_prompt_cache_dir"]
            except Exception:
                pass
            raise RuntimeError(f"Failed to enable persistent prompt for profile '{p.profile_id}': {e}") from e

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
        prompt = getattr(self, "_voice_clone_prompt", None) if bool(getattr(self, "_profile_prompt_enabled", False)) else None
        audio, _sr = self._runtime.generate_audio(
            text=str(text),
            language=str(self._language),
            # Even when a prompt is present, keep `instruct` enabled.
            #
            # Why: empirically, some OmniVoice configurations become unstable
            # (garbled/noisy output) when `voice_clone_prompt` is provided with
            # `instruct=None`. Keeping the style instruction preserves text
            # fidelity while the prompt anchors speaker identity.
            instruct=self._instruct,
            voice_clone_prompt=prompt,
            duration=getattr(self, "_duration_s", None),
            speed=None,
            settings=getattr(self, "_settings", OmniVoiceSettings()),
        )
        return np.asarray(audio, dtype=np.float32).reshape(-1)

    def synthesize_with_speed(self, text: str, speed: float) -> np.ndarray:
        sp = float(speed) if speed is not None else 1.0
        prompt = getattr(self, "_voice_clone_prompt", None) if bool(getattr(self, "_profile_prompt_enabled", False)) else None
        audio, _sr = self._runtime.generate_audio(
            text=str(text),
            language=str(self._language),
            instruct=self._instruct,
            voice_clone_prompt=prompt,
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
                "quality_preset": str(getattr(self, "_quality_preset", "standard") or "standard"),
                "language": str(getattr(self, "_language", "en") or "en"),
            }
        )
        try:
            info["runtime"] = dict(self._runtime.runtime_info())
        except Exception:
            pass
        return info

    def set_quality_preset(self, preset: str) -> bool:
        from ..quality_preset import normalize_quality_preset

        p = normalize_quality_preset(str(preset))
        self._quality_preset = str(p)

        # Map engine-agnostic presets to OmniVoice knobs.
        # Keep guidance_scale stable initially; steps are the main speed/quality driver.
        #
        # IMPORTANT:
        # OmniVoice can be *dramatically* slower on MPS in practice (see
        # `omnivoice/runtime.py` for rationale). Prefer conservative step counts
        # here; users can always raise them explicitly via engine params.
        if p == "low":
            self._settings.num_step = 8
            self._settings.guidance_scale = 2.0
        elif p == "standard":
            self._settings.num_step = 12
            self._settings.guidance_scale = 2.0
        else:
            self._settings.num_step = 24
            self._settings.guidance_scale = 2.0
        return True

    def get_quality_preset(self) -> str | None:
        return str(getattr(self, "_quality_preset", None) or "standard")

