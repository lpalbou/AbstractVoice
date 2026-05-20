"""TTS + voice/language methods for VoiceManager.

This module intentionally focuses on orchestration and keeps heavy engine details
behind adapters.
"""

from __future__ import annotations

from dataclasses import replace
import threading
import time

from ..adapters.base import TTSAdapter
from ..text_sanitize import sanitize_markdown_for_speech
from ..adapters.tts_registry import create_tts_adapter
from ..speech_request import SpeechCapabilities, SpeechCapability, build_speech_request

def _resolve_sanitize_syntax_arg(
    sanitize_syntax: bool,
    saninitze_syntax: bool | None,
) -> bool:
    """Resolve sanitize_syntax value, supporting a common misspelling alias.

    `saninitze_syntax` is accepted as an alias for backward-compat / typo tolerance.
    """
    resolved = bool(sanitize_syntax)
    if saninitze_syntax is not None:
        # If caller provided both, treat `saninitze_syntax` as an alias override,
        # but reject the one ambiguous/confusing case we can detect: opt-out
        # via the canonical flag + opt-in via the alias.
        if bool(sanitize_syntax) is False and bool(saninitze_syntax) is True:
            raise ValueError("Pass only one of sanitize_syntax or saninitze_syntax (alias).")
        resolved = bool(saninitze_syntax)

    return resolved


class TtsMixin:
    def _adapter_method_overridden(self, adapter, method_name: str) -> bool:
        if adapter is None:
            return False
        cls_method = getattr(type(adapter), method_name, None)
        base_method = getattr(TTSAdapter, method_name, None)
        return callable(cls_method) and cls_method is not base_method

    def _build_speech_request(
        self,
        text: str,
        *,
        voice: str | None = None,
        instructions: str | None = None,
        output_format: str | None = None,
        sanitize_syntax: bool = True,
        speed: float | None = None,
    ):
        request = build_speech_request(
            text,
            language=str(getattr(self, "language", None) or "en"),
            provider=(
                getattr(self, "_tts_engine_name", None)
                or getattr(self, "_tts_engine_preference", None)
            ),
            model=getattr(self, "tts_model", None),
            voice=voice,
            instructions=instructions,
            speed=speed,
            quality_preset=self.get_tts_quality_preset(),
            output_format=output_format,
            sanitize_syntax=bool(sanitize_syntax),
        )
        speech_text = str(request.text)
        if request.sanitize_syntax:
            speech_text = sanitize_markdown_for_speech(speech_text)
        if speech_text != request.text:
            request = replace(request, text=speech_text)
        return request

    def _set_last_tts_metrics(self, metrics: dict | None) -> None:
        lock = getattr(self, "_last_tts_metrics_lock", None)
        if lock is None:
            setattr(self, "_last_tts_metrics", metrics)
            return
        try:
            with lock:
                setattr(self, "_last_tts_metrics", metrics)
        except Exception:
            setattr(self, "_last_tts_metrics", metrics)

    def pop_last_tts_metrics(self) -> dict | None:
        lock = getattr(self, "_last_tts_metrics_lock", None)
        if lock is None:
            m = getattr(self, "_last_tts_metrics", None)
            setattr(self, "_last_tts_metrics", None)
            return m
        try:
            with lock:
                m = getattr(self, "_last_tts_metrics", None)
                setattr(self, "_last_tts_metrics", None)
                return m
        except Exception:
            m = getattr(self, "_last_tts_metrics", None)
            setattr(self, "_last_tts_metrics", None)
            return m

    def _get_voice_cloner(self):
        if getattr(self, "_voice_cloner", None) is None:
            try:
                from ..cloning import VoiceCloner
            except Exception as e:
                raise RuntimeError(
                    "Voice cloning is an optional feature.\n"
                    "Install with: pip install \"abstractvoice[cloning]\"\n"
                    f"Original error: {e}"
                ) from e

            # Use a slightly larger STT model for one-time reference-text auto-fallback.
            self._voice_cloner = VoiceCloner(
                debug=bool(getattr(self, "debug_mode", False)),
                whisper_model=getattr(self, "whisper_model", "tiny"),
                reference_text_whisper_model="small",
                allow_downloads=bool(getattr(self, "allow_downloads", True)),
                default_engine=str(getattr(self, "cloning_engine", "omnivoice") or "omnivoice"),
                remote_base_url=getattr(self, "remote_base_url", None),
                remote_api_key=getattr(self, "remote_api_key", None),
                remote_timeout_s=getattr(self, "remote_timeout_s", None),
                remote_tts_model=getattr(self, "tts_model", None),
            )
        return self._voice_cloner

    def clone_voice(
        self,
        reference_audio_path: str,
        name: str | None = None,
        *,
        reference_text: str | None = None,
        engine: str | None = None,
    ) -> str:
        return self._get_voice_cloner().clone_voice(
            reference_audio_path,
            name=name,
            reference_text=reference_text,
            engine=engine,
        )

    def clone_voice_from_wav_bytes(
        self,
        wav_bytes: bytes,
        name: str | None = None,
        *,
        reference_text: str | None = None,
        engine: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> str:
        """Create a new cloned voice from an in-memory WAV payload.

        This is the API surface used by client/server integrations where the
        reference audio arrives as uploaded bytes rather than a local file path.
        """
        cloner = self._get_voice_cloner()
        if hasattr(cloner, "clone_voice_from_wav_bytes"):
            return cloner.clone_voice_from_wav_bytes(
                wav_bytes,
                name=name,
                reference_text=reference_text,
                engine=engine,
                meta=meta,
            )
        # Backward-compatible fallback for older cloner versions (should not
        # normally be needed inside this repo).
        import tempfile
        from pathlib import Path

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = Path(tmp.name)
        try:
            tmp.write(wav_bytes)
            tmp.flush()
        finally:
            try:
                tmp.close()
            except Exception:
                pass
        try:
            return self.clone_voice(str(tmp_path), name=name, reference_text=reference_text, engine=engine)
        finally:
            try:
                tmp_path.unlink(missing_ok=True)  # type: ignore[arg-type]
            except Exception:
                pass

    def list_cloned_voices(self):
        return self._get_voice_cloner().list_cloned_voices()

    def get_cloned_voice(self, voice_id: str):
        return self._get_voice_cloner().get_cloned_voice(voice_id)

    def get_cloned_voice_store_dir(self) -> str:
        """Return the on-disk folder where cloned voices are stored."""
        try:
            return str(self._get_voice_cloner().get_store_base_dir())
        except Exception:
            return ""

    def set_cloned_voice_reference_text(self, voice_id: str, reference_text: str) -> bool:
        """Update a cloned voice's reference transcript (quality fix).

        A bad reference transcript commonly causes repeated/incorrect words in output.
        """
        self._get_voice_cloner().set_reference_text(voice_id, reference_text)
        return True

    def export_voice(self, voice_id: str, path: str) -> str:
        return self._get_voice_cloner().export_voice(voice_id, path)

    def import_voice(self, path: str) -> str:
        return self._get_voice_cloner().import_voice(path)

    def set_cloned_tts_quality(self, preset: str) -> bool:
        """Set cloned TTS quality preset: low|standard|high (aliases: fast, balanced)."""
        from ..quality_preset import normalize_quality_preset

        p = normalize_quality_preset(str(preset))
        self._get_voice_cloner().set_quality_preset(p)
        return True

    def get_cloned_tts_quality_preset(self) -> str | None:
        """Return the current cloned TTS quality preset (best-effort)."""
        try:
            cloner = self._get_voice_cloner()
        except Exception:
            return None
        try:
            if hasattr(cloner, "get_quality_preset"):
                return str(cloner.get_quality_preset() or "").strip() or None
        except Exception:
            return None
        return None

    def set_tts_quality_preset(self, preset: str) -> bool:
        """Set base TTS engine quality preset: low|standard|high (best-effort).

        This is an engine-agnostic knob. Engines that don't support quality tuning
        may ignore it and return False.
        """
        if not getattr(self, "tts_adapter", None):
            return False
        from ..quality_preset import normalize_quality_preset

        try:
            p = normalize_quality_preset(str(preset))
        except Exception:
            return False
        try:
            adapter = getattr(self, "tts_adapter", None)
            if adapter is None:
                return False
            provider_name = str(
                getattr(adapter, "engine_id", None)
                or getattr(adapter, "provider", None)
                or getattr(self, "_tts_engine_name", None)
                or getattr(self, "_tts_engine_preference", None)
                or ""
            ).strip().lower() or None
            model_name = str(
                getattr(adapter, "model_id", None)
                or getattr(self, "tts_model", None)
                or ""
            ).strip() or None
            bytes_support = self._compatibility_support_level(
                kind="tts",
                provider=provider_name,
                model=model_name,
                surface="bytes",
                feature="quality_preset",
            )
            playback_support = self._compatibility_support_level(
                kind="tts",
                provider=provider_name,
                model=model_name,
                surface="playback",
                feature="quality_preset",
            )
            if bytes_support == "unsupported" and playback_support == "unsupported":
                return False
            if hasattr(adapter, "set_quality_preset"):
                return bool(adapter.set_quality_preset(p))
        except Exception:
            return False
        return False

    def get_tts_quality_preset(self) -> str | None:
        """Return the current base TTS quality preset, if supported."""
        adapter = getattr(self, "tts_adapter", None)
        if adapter is None:
            return None
        try:
            if hasattr(adapter, "get_quality_preset"):
                return adapter.get_quality_preset()
        except Exception:
            return None
        return None

    def get_compatibility_catalog(self):
        """Return the central provider/model compatibility catalog."""
        from ..compatibility import build_compatibility_catalog

        adapter = getattr(self, "tts_adapter", None)
        current_tts_provider = str(
            getattr(adapter, "engine_id", None)
            or getattr(adapter, "provider", None)
            or getattr(self, "_tts_engine_name", None)
            or getattr(self, "_tts_engine_preference", None)
            or ""
        ).strip().lower() or None
        current_tts_model = None
        if adapter is not None:
            current_tts_model = getattr(adapter, "model_id", None)
        if not isinstance(current_tts_model, str) or not current_tts_model.strip():
            current_tts_model = getattr(self, "tts_model", None)

        stt_adapter = getattr(self, "stt_adapter", None)
        current_stt_provider = str(
            getattr(stt_adapter, "engine_id", None)
            or getattr(stt_adapter, "provider", None)
            or getattr(self, "_stt_engine_name", None)
            or getattr(self, "_stt_engine_preference", None)
            or getattr(self, "stt_engine", None)
            or ""
        ).strip().lower() or None
        current_stt_model = (
            getattr(stt_adapter, "model_id", None)
            or getattr(self, "stt_model", None)
        )

        current_remote_tts_model = None
        cloner = getattr(self, "_voice_cloner", None)
        if cloner is not None:
            current_remote_tts_model = getattr(cloner, "_remote_tts_model", None)
        current_cloning_provider = str(getattr(self, "cloning_engine", None) or "").strip().lower() or None
        if (
            (not isinstance(current_remote_tts_model, str) or not current_remote_tts_model.strip())
            and current_cloning_provider in {"openai", "openai-compatible"}
        ):
            current_remote_tts_model = getattr(self, "tts_model", None)

        return build_compatibility_catalog(
            current_tts_provider=current_tts_provider,
            current_tts_model=(str(current_tts_model).strip() if isinstance(current_tts_model, str) and current_tts_model.strip() else None),
            current_stt_provider=current_stt_provider,
            current_stt_model=(str(current_stt_model).strip() if isinstance(current_stt_model, str) and current_stt_model.strip() else None),
            current_cloning_provider=current_cloning_provider,
            current_remote_tts_model=(
                str(current_remote_tts_model).strip()
                if isinstance(current_remote_tts_model, str) and current_remote_tts_model.strip()
                else None
            ),
        )

    def get_capability_support(
        self,
        *,
        kind: str,
        feature: str,
        provider: str | None = None,
        model: str | None = None,
        surface: str = "default",
    ) -> dict | None:
        """Return support metadata for one feature, if known."""
        normalized_kind = str(kind or "").strip().lower()
        provider_name = str(provider or "").strip().lower()
        model_name = str(model).strip() if isinstance(model, str) and model.strip() else None

        if not provider_name and normalized_kind == "tts":
            adapter = getattr(self, "tts_adapter", None)
            provider_name = str(
                getattr(adapter, "engine_id", None)
                or getattr(adapter, "provider", None)
                or getattr(self, "_tts_engine_name", None)
                or getattr(self, "_tts_engine_preference", None)
                or ""
            ).strip().lower()
            if model_name is None:
                model_name = str(
                    getattr(adapter, "model_id", None)
                    or getattr(self, "tts_model", None)
                    or ""
                ).strip() or None
        elif not provider_name and normalized_kind == "stt":
            adapter = getattr(self, "stt_adapter", None)
            provider_name = str(
                getattr(adapter, "engine_id", None)
                or getattr(adapter, "provider", None)
                or getattr(self, "_stt_engine_name", None)
                or getattr(self, "_stt_engine_preference", None)
                or getattr(self, "stt_engine", None)
                or ""
            ).strip().lower().replace("_", "-")
            if model_name is None:
                model_name = str(
                    getattr(adapter, "model_id", None)
                    or getattr(self, "stt_model", None)
                    or ""
                ).strip() or None
        elif not provider_name and normalized_kind == "cloning":
            provider_name = str(getattr(self, "cloning_engine", None) or "").strip().lower().replace("-", "_")
            if model_name is None and provider_name in {"openai", "openai_compatible"}:
                cloner = getattr(self, "_voice_cloner", None)
                model_name = str(
                    getattr(cloner, "_remote_tts_model", None)
                    or getattr(self, "tts_model", None)
                    or ""
                ).strip() or None

        catalog = self.get_compatibility_catalog()
        support = catalog.support_for(
            kind=normalized_kind,
            provider=provider_name,
            model=model_name,
            surface=str(surface or "default").strip() or "default",
            feature=str(feature or "").strip(),
        )
        return support.to_dict() if support is not None else None

    def _compatibility_support_level(
        self,
        *,
        kind: str,
        feature: str,
        provider: str | None = None,
        model: str | None = None,
        surface: str = "default",
    ) -> str:
        support = self.get_capability_support(
            kind=kind,
            feature=feature,
            provider=provider,
            model=model,
            surface=surface,
        )
        if isinstance(support, dict):
            return str(support.get("support") or "unsupported")
        return "unsupported"

    def _resolve_clone_engine_name(self, cloner, voice_id: str) -> str | None:
        try:
            info = cloner.get_cloned_voice(str(voice_id)) or {}
        except Exception:
            info = {}
        return str(info.get("engine") or "").strip().lower() or None

    def _effective_clone_speed(
        self,
        *,
        cloner,
        voice_id: str,
        surface: str,
    ) -> tuple[float, str | None]:
        clone_engine_name = self._resolve_clone_engine_name(cloner, voice_id)
        clone_speed = float(getattr(self, "speed", 1.0) or 1.0)
        if clone_engine_name:
            speed_support = self._compatibility_support_level(
                kind="cloning",
                provider=clone_engine_name,
                model=(str(getattr(cloner, "_remote_tts_model", "")).strip() or None),
                surface=str(surface or "speak_bytes").strip() or "speak_bytes",
                feature="speed",
            )
            if speed_support == "unsupported":
                clone_speed = 1.0
        return clone_speed, clone_engine_name

    def find_compatible_models(
        self,
        *,
        kind: str,
        feature: str,
        surface: str = "default",
        support_in: tuple[str, ...] = ("native", "emulated", "conditional"),
    ) -> list[dict]:
        """Return provider/model pairs that support a feature."""
        catalog = self.get_compatibility_catalog()
        return catalog.find_models(
            kind=str(kind or "").strip().lower(),
            feature=str(feature or "").strip(),
            surface=str(surface or "default").strip() or "default",
            support_in=tuple(str(item) for item in tuple(support_in or ())),
        )

    def get_tts_capabilities(self, *, surface: str = "bytes") -> SpeechCapabilities:
        """Return package-owned TTS control support for the active engine."""
        adapter = getattr(self, "tts_adapter", None)
        try:
            engine_id = str(
                getattr(adapter, "engine_id", "")
                or getattr(adapter, "provider", "")
                or getattr(self, "_tts_engine_name", "")
                or getattr(self, "_tts_engine_preference", "")
                or ""
            ).strip().lower()
        except Exception:
            engine_id = ""
        model_id = None
        try:
            model_id = getattr(adapter, "model_id", None) or getattr(self, "tts_model", None)
        except Exception:
            model_id = getattr(self, "tts_model", None)

        from ..compatibility import TTS_COMPATIBILITY_FEATURES

        catalog = self.get_compatibility_catalog()
        fields: dict[str, SpeechCapability] = {}
        for feature_name in TTS_COMPATIBILITY_FEATURES:
            support = catalog.support_for(
                kind="tts",
                provider=engine_id or "tts",
                model=(str(model_id).strip() if isinstance(model_id, str) and model_id.strip() else None),
                surface=str(surface or "bytes").strip() or "bytes",
                feature=str(feature_name),
            )
            if support is None:
                fields[str(feature_name)] = SpeechCapability(
                    name=str(feature_name),
                    support="unsupported",
                    reason="The active provider/model does not declare support for this field.",
                )
                continue
            fields[str(feature_name)] = SpeechCapability(
                name=str(feature_name),
                support=str(support.support),
                reason=support.reason,
            )
        return SpeechCapabilities(fields=fields)

    # ------------------------------------------------------------------
    # Audio delivery mode (buffered vs streamed)
    # ------------------------------------------------------------------

    def set_tts_delivery_mode(self, mode: str | None) -> bool:
        """Set audio delivery mode for both base TTS and cloned voices.

        - buffered: synthesize full audio first (smooth playback)
        - streamed: enqueue audio chunks progressively when available (lower TTFB)

        This is an override. When set, it supersedes legacy `cloned_tts_streaming`.
        """
        if mode is None:
            setattr(self, "tts_delivery_mode", None)
            return True
        from ..tts.delivery_mode import normalize_audio_delivery_mode

        m = normalize_audio_delivery_mode(mode)
        setattr(self, "tts_delivery_mode", str(m))
        # Keep legacy clone toggle aligned for back-compat introspection.
        try:
            setattr(self, "cloned_tts_streaming", bool(m == "streamed"))
        except Exception:
            pass
        return True

    def get_tts_delivery_modes(self) -> dict:
        """Return effective delivery modes (base vs clone) for debugging/UX."""
        override = None
        try:
            override = getattr(self, "tts_delivery_mode", None)
        except Exception:
            override = None
        override = str(override).strip().lower() if override else None

        # Base TTS defaults to buffered when no override is set.
        base_mode = "streamed" if override == "streamed" else "buffered"

        clone_streaming = bool(getattr(self, "cloned_tts_streaming", True))
        if override in ("buffered", "streamed"):
            clone_streaming = bool(override == "streamed")
        clone_mode = "streamed" if clone_streaming else "buffered"

        return {"override": override, "base": base_mode, "clone": clone_mode}

    def get_tts_delivery_mode(self) -> str:
        """Return a single summary string (buffered|streamed|mixed)."""
        modes = self.get_tts_delivery_modes()
        b = str(modes.get("base") or "")
        c = str(modes.get("clone") or "")
        if b == c and b in ("buffered", "streamed"):
            return b
        return "mixed"

    def get_cloning_runtime_info(self):
        return self._get_voice_cloner().get_runtime_info()

    def preload_cloning_engine(
        self,
        *,
        engine: str | None = None,
        voice: str | None = None,
        language: str | None = None,
        speed: float | None = None,
    ) -> dict:
        cloner = self._get_voice_cloner()
        return dict(
            cloner.preload_engine(
                engine=engine,
                voice_id=voice,
                language=str(language or getattr(self, "language", None) or "en"),
                speed=float(speed if speed is not None else getattr(self, "speed", 1.0) or 1.0),
            )
        )

    def list_resident_components(self) -> list[dict]:
        cloner = getattr(self, "_voice_cloner", None)
        if cloner is None:
            return []
        try:
            return [dict(item) for item in list(cloner.list_loaded_engines() or [])]
        except Exception:
            return []

    def rename_cloned_voice(self, voice_id: str, new_name: str) -> bool:
        self._get_voice_cloner().rename_cloned_voice(voice_id, new_name)
        return True

    def delete_cloned_voice(self, voice_id: str) -> bool:
        self._get_voice_cloner().delete_cloned_voice(voice_id)
        return True

    def unload_cloning_engines(self, *, keep_engine: str | None = None) -> int:
        """Best-effort free memory held by loaded cloning engines.

        This is critical for large backends (e.g. Chroma). It does NOT delete any
        cloned voices; it only releases in-memory model weights.
        """
        cloner = getattr(self, "_voice_cloner", None)
        if cloner is None:
            return 0
        try:
            if keep_engine:
                return int(cloner.unload_engines_except(str(keep_engine)))
            return int(cloner.unload_all_engines())
        except Exception:
            return 0

    def unload_cloning_engine(self, *, engine: str | None = None) -> dict:
        cloner = getattr(self, "_voice_cloner", None)
        if cloner is None:
            return {
                "engine": str(engine or getattr(self, "cloning_engine", None) or "").strip().lower() or None,
                "resident": False,
                "unloaded": False,
                "state": "not_loaded",
                "local": True,
                "unloadable": True,
            }
        name = str(engine or getattr(self, "cloning_engine", None) or "").strip().lower() or None
        unloaded = False
        if name:
            try:
                unloaded = bool(cloner.unload_engine(name))
            except Exception:
                unloaded = False
        return {
            "engine": name,
            "resident": False,
            "unloaded": bool(unloaded),
            "state": "unloaded" if unloaded else "not_loaded",
            "local": True,
            "unloadable": True,
        }

    def unload_piper_voice(self) -> bool:
        """Best-effort release of Piper voice weights/session (keeps audio output ready).

        This helps reduce memory pressure when switching to large cloning backends.
        """
        try:
            adapter = getattr(self, "tts_adapter", None)
            if adapter is None:
                return False
            if hasattr(adapter, "unload"):
                adapter.unload()
                return True
            # Back-compat: drop voice object if present.
            if hasattr(adapter, "_voice"):
                setattr(adapter, "_voice", None)
                return True
        except Exception:
            return False
        return False

    def _tts_adapter_unavailable_message(self, adapter=None) -> str:
        adapter = adapter if adapter is not None else getattr(self, "tts_adapter", None)
        try:
            get_reason = getattr(adapter, "get_unavailable_reason", None)
            reason = get_reason() if callable(get_reason) else None
            if reason:
                return str(reason)
        except Exception:
            pass
        engine = ""
        try:
            engine = str(getattr(adapter, "engine_id", "") or "").strip().lower()
        except Exception:
            engine = ""
        if not engine:
            try:
                engine = str(getattr(self, "_tts_engine_name", "") or "").strip().lower()
            except Exception:
                engine = ""
        if not engine:
            try:
                engine = str(getattr(self, "_tts_engine_preference", "") or "").strip().lower()
            except Exception:
                engine = ""
        if engine:
            return f"TTS engine '{engine}' is not available."
        return "No TTS adapter available"

    def reset_tts_profile(self, *, language: str | None = None):
        """Reset the active TTS adapter to its engine/language default profile."""
        adapter = getattr(self, "tts_adapter", None)
        if adapter is None:
            return None
        lang = str(language or getattr(self, "language", None) or "en").strip().lower() or "en"
        try:
            reset = getattr(adapter, "reset_profile", None)
            if callable(reset):
                return reset(language=lang)
        except Exception:
            pass

        default_id = None
        try:
            getter = getattr(adapter, "get_default_profile_id", None)
            default_id = getter(lang) if callable(getter) else None
        except Exception:
            default_id = None
        if default_id:
            try:
                adapter.set_profile(str(default_id))
            except Exception:
                pass
        try:
            return adapter.get_active_profile()
        except Exception:
            return None

    def set_tts_engine(
        self,
        engine: str,
        *,
        tts_model: str | None = None,
        allow_downloads: bool | None = None,
        auto_load: bool = True,
    ) -> str:
        """Switch the base TTS engine and reset to that engine's default profile."""
        requested = str(engine or "").strip().lower().replace("_", "-") or "auto"
        if requested in ("remote", "compatible", "proxy"):
            requested = "openai-compatible"

        old_engine = getattr(self, "tts_engine", None)
        old_adapter = getattr(self, "tts_adapter", None)

        try:
            self.stop_speaking()
        except Exception:
            pass

        model_id = tts_model if tts_model is not None else getattr(self, "tts_model", None)
        adapter, resolved_engine = create_tts_adapter(
            engine=requested,
            language=str(getattr(self, "language", "en") or "en"),
            allow_downloads=bool(getattr(self, "allow_downloads", True) if allow_downloads is None else allow_downloads),
            auto_load=bool(auto_load),
            debug_mode=bool(getattr(self, "debug_mode", False)),
            model_id=model_id,
            base_url=getattr(self, "remote_base_url", None),
            api_key=getattr(self, "remote_api_key", None),
            timeout_s=getattr(self, "remote_timeout_s", None),
        )
        if adapter is None:
            raise RuntimeError(f"TTS engine '{requested}' is not available in this environment.")

        target_language = str(getattr(self, "language", "en") or "en").strip().lower() or "en"
        try:
            supported = list(adapter.get_supported_languages() or [])
        except Exception:
            supported = []
        language_changed = False
        if supported and target_language not in supported:
            fallback_language = "en" if "en" in supported else str(supported[0])
            try:
                adapter.set_language(str(fallback_language))
            except Exception:
                pass
            self.language = str(fallback_language)
            language_changed = str(fallback_language) != target_language
        if language_changed and getattr(self, "voice_recognizer", None):
            try:
                self.voice_recognizer.stop()
            except Exception:
                pass
            self.voice_recognizer = None

        from ..tts.adapter_tts_engine import AdapterTTSEngine

        new_engine = AdapterTTSEngine(adapter, debug_mode=bool(getattr(self, "debug_mode", False)))
        self.tts_adapter = adapter
        self.tts_engine = new_engine
        self._tts_engine_name = str(resolved_engine)
        self._tts_engine_preference = str(requested)
        if tts_model is not None:
            self.tts_model = tts_model
        self.reset_tts_profile(language=str(getattr(self, "language", "en") or "en"))
        self._wire_tts_callbacks()

        if old_engine is not None and old_engine is not new_engine:
            try:
                if hasattr(old_engine, "cleanup"):
                    old_engine.cleanup()
                elif hasattr(old_engine, "audio_player") and old_engine.audio_player:
                    old_engine.audio_player.cleanup()
            except Exception:
                pass
        _ = old_adapter
        return str(resolved_engine)

    def speak(
        self,
        text,
        speed=1.0,
        callback=None,
        voice: str | None = None,
        *,
        sanitize_syntax: bool = True,
        saninitze_syntax: bool | None = None,
    ):
        sp = speed if speed != 1.0 else self.speed
        if not self.tts_engine:
            raise RuntimeError("No TTS engine available")

        request = self._build_speech_request(
            str(text),
            voice=voice,
            sanitize_syntax=_resolve_sanitize_syntax_arg(sanitize_syntax, saninitze_syntax),
            speed=float(sp if sp is not None else 1.0),
        )
        speak_text = str(request.text)

        # ------------------------------------------------------------------
        # Delivery-mode override: streamed vs buffered (engine-agnostic)
        # ------------------------------------------------------------------
        # When `tts_delivery_mode` is set, we apply it consistently to:
        # - base TTS playback (where supported)
        # - cloned voice playback (overrides legacy `cloned_tts_streaming`)
        delivery_mode = None
        try:
            delivery_mode = getattr(self, "tts_delivery_mode", None)
        except Exception:
            delivery_mode = None
        delivery_mode = str(delivery_mode).strip().lower() if delivery_mode else None

        # ------------------------------------------------------------------
        # Base TTS: optional streamed delivery (chunked playback)
        # ------------------------------------------------------------------
        if not voice and delivery_mode == "streamed":
            # Speed control for base TTS often uses post-processing time-stretch
            # on the *full* waveform. Doing that per-chunk can introduce artifacts.
            # If speed is requested, keep the robust buffered path for now.
            try:
                sp_f = float(sp or 1.0)
            except Exception:
                sp_f = 1.0

            if sp_f == 1.0:
                # Delegate to the shared streaming bridge so stop/pause/metrics and
                # LLM→TTS pipelining share one implementation.
                try:
                    stream = self.open_tts_text_stream(
                        voice=None,
                        callback=callback,
                        sanitize_syntax=False,  # `speak_text` already reflects sanitize_syntax.
                    )
                    stream.push(str(speak_text))
                    stream.close()
                    return True
                except Exception:
                    # Fall back to buffered speak() below.
                    pass

        # Optional cloned voice playback:
        # - stream chunks to the player for better perceived latency
        # - support cancellation on stop_speaking() / new input (best-effort)
        if voice:
            import numpy as np

            from ..audio.resample import linear_resample_mono

            # Clear prior metrics for this new utterance.
            self._set_last_tts_metrics(None)

            # Stop any current speech and reset cancel token.
            try:
                self.stop_speaking()
            except Exception:
                pass

            # IMPORTANT: cancellation must be per-utterance.
            # If we reuse/clear the same Event, an old synthesis thread could resume
            # after a new request starts (race), causing "old audio" to continue.
            try:
                old = getattr(self, "_cloned_cancel_event", None)
                if old is not None:
                    old.set()
            except Exception:
                pass
            cancel = threading.Event()
            setattr(self, "_cloned_cancel_event", cancel)

            cloner = self._get_voice_cloner()
            # Prefer playing cloned audio at its native rate (F5 is typically 24kHz).
            target_sr = 24000
            sp, clone_engine_name = self._effective_clone_speed(
                cloner=cloner,
                voice_id=str(voice),
                surface="speak_bytes",
            )

            def _worker():
                try:
                    synth_active = getattr(self, "_cloned_synthesis_active", None)
                    if synth_active is not None:
                        try:
                            synth_active.set()
                        except Exception:
                            pass

                    # Option: generate full audio first (smooth playback) vs streaming (faster TTFB).
                    clone_streaming = bool(getattr(self, "cloned_tts_streaming", True))
                    if delivery_mode in ("buffered", "streamed"):
                        clone_streaming = bool(delivery_mode == "streamed")

                    if not bool(clone_streaming):
                        import io
                        import soundfile as sf

                        t0 = time.monotonic()
                        wav_bytes = cloner.speak_to_bytes(
                            str(speak_text),
                            voice_id=voice,
                            format="wav",
                            speed=sp,
                            language=str(getattr(self, "language", None) or "en"),
                        )
                        t1 = time.monotonic()
                        if cancel.is_set():
                            return
                        audio, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=True)
                        mono = np.mean(audio, axis=1).astype(np.float32).reshape(-1)
                        sr = int(sr)

                        try:
                            audio_samples = int(len(mono))
                        except Exception:
                            audio_samples = 0
                        audio_s = (float(audio_samples) / float(sr)) if sr and audio_samples else 0.0
                        synth_s = float(t1 - t0)
                        self._set_last_tts_metrics(
                            {
                                "engine": "clone",
                                "clone_engine": clone_engine_name or None,
                                "voice_id": str(voice),
                                "streaming": False,
                                "synth_s": synth_s,
                                "audio_s": float(audio_s),
                                "rtf": (synth_s / float(audio_s)) if audio_s else None,
                                "sample_rate": int(sr) if sr else None,
                                "audio_samples": int(audio_samples),
                                "ts": time.time(),
                            }
                        )

                        if hasattr(self.tts_engine, "begin_playback"):
                            self.tts_engine.begin_playback(callback=callback, sample_rate=sr)
                        if cancel.is_set():
                            return
                        if hasattr(self.tts_engine, "enqueue_audio"):
                            try:
                                self.tts_engine.enqueue_audio(mono, sample_rate=sr)
                            except TypeError:
                                self.tts_engine.enqueue_audio(mono)
                        elif hasattr(self.tts_engine, "audio_player") and self.tts_engine.audio_player:
                            try:
                                self.tts_engine.audio_player.play_audio(mono, sample_rate=sr)
                            except TypeError:
                                self.tts_engine.audio_player.play_audio(mono)
                        return

                    # Streaming path: fewer, larger batches reduce audible cuts and overhead.
                    t0 = time.monotonic()
                    first_chunk_t = None
                    total_samples = 0
                    chunks = 0
                    chunks_iter = cloner.speak_to_audio_chunks(
                        str(speak_text),
                        voice_id=voice,
                        speed=sp,
                        # Smaller batches reduce time-to-first-audio.
                        # When streaming is enabled via `tts_delivery_mode`, prefer more
                        # frequent phrase breaks even within a single long sentence.
                        max_chars=(120 if str(delivery_mode or "") == "streamed" else 240),
                        language=str(getattr(self, "language", None) or "en"),
                    )

                    # Begin a playback session once (so TTS lifecycle hooks are correct).
                    if hasattr(self.tts_engine, "begin_playback"):
                        self.tts_engine.begin_playback(callback=callback, sample_rate=target_sr)

                    for chunk, sr in chunks_iter:
                        if cancel.is_set():
                            break
                        if first_chunk_t is None:
                            first_chunk_t = time.monotonic()
                        mono = np.asarray(chunk, dtype=np.float32).reshape(-1)
                        if int(sr) != target_sr:
                            mono = linear_resample_mono(mono, int(sr), target_sr)
                        try:
                            total_samples += int(len(mono))
                            chunks += 1
                        except Exception:
                            pass

                        if hasattr(self.tts_engine, "enqueue_audio"):
                            try:
                                self.tts_engine.enqueue_audio(mono, sample_rate=target_sr)
                            except TypeError:
                                self.tts_engine.enqueue_audio(mono)
                        elif hasattr(self.tts_engine, "audio_player") and self.tts_engine.audio_player:
                            try:
                                self.tts_engine.audio_player.play_audio(mono, sample_rate=target_sr)
                            except TypeError:
                                self.tts_engine.audio_player.play_audio(mono)
                        else:
                            break

                    t1 = time.monotonic()
                    audio_s = (float(total_samples) / float(target_sr)) if total_samples else 0.0
                    synth_s = float(t1 - t0)
                    ttfb_s = (float(first_chunk_t - t0) if first_chunk_t is not None else None)
                    self._set_last_tts_metrics(
                        {
                            "engine": "clone",
                            "clone_engine": clone_engine_name or None,
                            "voice_id": str(voice),
                            "streaming": True,
                            "cancelled": bool(cancel.is_set()),
                            "synth_s": synth_s,
                            "ttfb_s": ttfb_s,
                            "audio_s": float(audio_s),
                            "rtf": (synth_s / float(audio_s)) if audio_s else None,
                            "sample_rate": int(target_sr),
                            "audio_samples": int(total_samples),
                            "chunks": int(chunks),
                            "ts": time.time(),
                        }
                    )
                except Exception as e:
                    # Best-effort: never crash caller thread.
                    try:
                        self._set_last_tts_metrics(
                            {
                                "engine": "clone",
                                "clone_engine": clone_engine_name or None,
                                "voice_id": str(voice),
                                "error": str(e),
                                "ts": time.time(),
                            }
                        )
                    except Exception:
                        pass
                    if bool(getattr(self, "debug_mode", False)):
                        print(f"⚠️  Cloned TTS failed: {e}")
                finally:
                    try:
                        synth_active = getattr(self, "_cloned_synthesis_active", None)
                        if synth_active is not None:
                            synth_active.clear()
                    except Exception:
                        pass

            threading.Thread(target=_worker, daemon=True).start()
            return True

        ok = self.tts_engine.speak(speak_text, sp, callback)
        # Mirror adapter metrics into the manager for a single "last TTS metrics"
        # source of truth (used by the verbose REPL).
        try:
            m = getattr(self.tts_engine, "last_tts_metrics", None)
            if isinstance(m, dict) and m:
                self._set_last_tts_metrics(dict(m))
        except Exception:
            pass
        return ok

    # Network/headless-friendly methods
    def speak_to_audio_chunks(
        self,
        text: str,
        *,
        voice: str | None = None,
        sanitize_syntax: bool = True,
        saninitze_syntax: bool | None = None,
    ):
        """Synthesize to an iterator of `(audio_chunk, sample_rate)` tuples.

        This is the engine-agnostic streaming surface for integrations that want
        incremental delivery (e.g. future HTTP chunked/streaming endpoints).

        Notes:
        - For base TTS, engines may yield a single chunk (buffered synthesis).
        - For cloned voices, chunking is implemented by cloning engines and is
          typically sentence/punctuation batch based.
        """
        # Clear prior metrics for this new utterance (best-effort; populated on completion).
        self._set_last_tts_metrics(None)

        speak_text = str(text)
        if _resolve_sanitize_syntax_arg(sanitize_syntax, saninitze_syntax):
            speak_text = sanitize_markdown_for_speech(speak_text)

        if voice:
            cloner = self._get_voice_cloner()

            clone_speed, clone_engine_name = self._effective_clone_speed(
                cloner=cloner,
                voice_id=str(voice),
                surface="speak_bytes",
            )

            def _gen_clone():
                import numpy as np

                t0 = time.monotonic()
                first_chunk_t = None
                chunks = 0
                total_audio_s = 0.0
                try:
                    for chunk, sr in cloner.speak_to_audio_chunks(
                        str(speak_text),
                        voice_id=str(voice),
                        speed=clone_speed,
                        max_chars=120,
                        language=str(getattr(self, "language", None) or "en"),
                    ):
                        mono = np.asarray(chunk, dtype=np.float32).reshape(-1)
                        if mono.size <= 0:
                            continue
                        sr_i = int(sr) if sr else 0
                        if sr_i > 0:
                            total_audio_s += float(len(mono)) / float(sr_i)
                        chunks += 1
                        if first_chunk_t is None:
                            first_chunk_t = time.monotonic()
                        yield mono, int(sr_i)
                finally:
                    t1 = time.monotonic()
                    synth_s = float(t1 - t0)
                    ttfb_s = float(first_chunk_t - t0) if first_chunk_t is not None else None
                    metrics = {
                        "engine": "clone",
                        "clone_engine": clone_engine_name,
                        "voice_id": str(voice),
                        "streaming": True,
                        "synth_s": synth_s,
                        "ttfb_s": ttfb_s,
                        "audio_s": float(total_audio_s),
                        "rtf": (synth_s / float(total_audio_s)) if total_audio_s else None,
                        "chunks": int(chunks),
                        "language": str(getattr(self, "language", None) or "en"),
                        "speed": float(clone_speed),
                        "ts": time.time(),
                    }
                    self._set_last_tts_metrics(metrics)

            return _gen_clone()

        # Base TTS chunks (best-effort).
        adapter = getattr(self, "tts_adapter", None)
        if adapter is None or (hasattr(adapter, "is_available") and not bool(adapter.is_available())):
            raise RuntimeError(self._tts_adapter_unavailable_message(adapter))

        engine_id = ""
        try:
            engine_id = str(getattr(adapter, "engine_id", "") or "").strip().lower()
        except Exception:
            engine_id = ""
        engine_id = engine_id or "tts"

        def _gen_base():
            import numpy as np

            t0 = time.monotonic()
            first_chunk_t = None
            chunks = 0
            total_audio_s = 0.0
            try:
                from ..tts.text_chunking import TextStreamChunker, TextStreamChunkingConfig

                try:
                    max_chars = int(getattr(adapter, "get_max_chars", lambda: 240)() or 240)
                except Exception:
                    max_chars = 240
                if not (int(max_chars) > 0):
                    max_chars = 240

                # Streaming-friendly segmentation (commas + sentence ends + hard cap).
                chunker = TextStreamChunker(config=TextStreamChunkingConfig(max_chars=int(max_chars), min_chars=1))
                segments = chunker.push(str(speak_text)) + chunker.flush()
                if not segments:
                    segments = [str(speak_text)]
                for seg_text in segments:
                    seg_text = str(seg_text or "").strip()
                    if not seg_text:
                        continue
                    for chunk, sr in adapter.synthesize_to_audio_chunks(str(seg_text)):
                        mono = np.asarray(chunk, dtype=np.float32).reshape(-1)
                        if mono.size <= 0:
                            continue
                        sr_i = int(sr) if sr else 0
                        if sr_i > 0:
                            total_audio_s += float(len(mono)) / float(sr_i)
                        chunks += 1
                        if first_chunk_t is None:
                            first_chunk_t = time.monotonic()
                        yield mono, int(sr_i)
            finally:
                t1 = time.monotonic()
                synth_s = float(t1 - t0)
                ttfb_s = float(first_chunk_t - t0) if first_chunk_t is not None else None
                metrics = {
                    "engine": engine_id,
                    "streaming": True,
                    "synth_s": synth_s,
                    "ttfb_s": ttfb_s,
                    "audio_s": float(total_audio_s),
                    "rtf": (synth_s / float(total_audio_s)) if total_audio_s else None,
                    "chunks": int(chunks),
                    "language": str(getattr(self, "language", None) or "en"),
                    "speed": float(getattr(self, "speed", 1.0) or 1.0),
                    "ts": time.time(),
                }
                self._set_last_tts_metrics(metrics)

        return _gen_base()

    def open_tts_text_stream(
        self,
        *,
        voice: str | None = None,
        callback=None,
        sanitize_syntax: bool = True,
        max_chars: int | None = None,
        min_chars: int | None = None,
    ):
        """Open a push-based streaming text -> TTS playback bridge.

        This is the intended abstraction for linking:
        - an LLM streaming response (text deltas)
        - into streamed TTS output (audio chunks)
        """
        if not getattr(self, "tts_engine", None):
            raise RuntimeError("No TTS engine available")

        # Clear prior metrics for this new utterance/stream.
        self._set_last_tts_metrics(None)

        # Stop any current speech and reset cancel token.
        try:
            self.stop_speaking()
        except Exception:
            pass

        try:
            old = getattr(self, "_cloned_cancel_event", None)
            if old is not None:
                old.set()
        except Exception:
            pass
        cancel = threading.Event()
        setattr(self, "_cloned_cancel_event", cancel)

        from ..tts.text_chunking import TextStreamChunkingConfig
        from ..tts.text_to_speech_stream import TextToSpeechStream, TextToSpeechStreamConfig

        # Chunking config.
        mc = int(max_chars) if isinstance(max_chars, int) and int(max_chars) > 0 else None
        mn = int(min_chars) if isinstance(min_chars, int) and int(min_chars) >= 0 else None

        adapter = getattr(self, "tts_adapter", None)
        engine_id = ""
        try:
            engine_id = str(getattr(adapter, "engine_id", "") or "").strip().lower()
        except Exception:
            engine_id = ""
        engine_id = engine_id or "tts"

        # If caller didn't specify max_chars, pick a sensible default.
        #
        # - base TTS: adapter controls segment size (quality/perf trade-off)
        # - cloned voices: prefer smaller segments by default to reduce time-to-first-audio
        if mc is None:
            if voice:
                mc = 120
            else:
                try:
                    mc = int(getattr(adapter, "get_max_chars", lambda: 240)() or 240)
                except Exception:
                    mc = 240
            if not (int(mc) > 0):
                mc = 240
        if mn is None:
            mn = 1

        chunk_cfg = TextStreamChunkingConfig(max_chars=int(mc), min_chars=int(mn))
        cfg = TextToSpeechStreamConfig(chunking=chunk_cfg)

        # Playback: begin only when we have the first audio chunk.
        started = {"v": False}

        def _on_audio_chunk(mono, sr: int) -> None:
            if cancel.is_set():
                return
            if not started["v"]:
                started["v"] = True
                try:
                    if hasattr(self.tts_engine, "begin_playback"):
                        self.tts_engine.begin_playback(callback=callback, sample_rate=(int(sr) if int(sr) > 0 else None))
                except Exception:
                    pass
            try:
                if hasattr(self.tts_engine, "enqueue_audio"):
                    self.tts_engine.enqueue_audio(mono, sample_rate=(int(sr) if int(sr) > 0 else None))
                else:
                    # Back-compat fallback: play_audio_array may exist.
                    play = getattr(self.tts_engine, "play_audio_array", None)
                    if callable(play):
                        play(mono, callback=None)
            except TypeError:
                self.tts_engine.enqueue_audio(mono)  # type: ignore[attr-defined]

        def _is_paused() -> bool:
            try:
                return bool(self.tts_engine.is_paused())
            except Exception:
                return False

        # Segment -> audio chunks.
        clone_engine_name = None
        if voice:
            cloner = self._get_voice_cloner()
            clone_speed, clone_engine_name = self._effective_clone_speed(
                cloner=cloner,
                voice_id=str(voice),
                surface="speak_bytes",
            )

            def _iter_chunks(seg_text: str):
                txt = str(seg_text or "")
                if sanitize_syntax:
                    txt = sanitize_markdown_for_speech(txt)
                return cloner.speak_to_audio_chunks(
                    txt,
                    voice_id=str(voice),
                    speed=clone_speed,
                    max_chars=int(mc or 240),
                    language=str(getattr(self, "language", None) or "en"),
                )
        else:
            if adapter is None or (hasattr(adapter, "is_available") and not bool(adapter.is_available())):
                raise RuntimeError(self._tts_adapter_unavailable_message(adapter))

            def _iter_chunks(seg_text: str):
                txt = str(seg_text or "")
                if sanitize_syntax:
                    txt = sanitize_markdown_for_speech(txt)
                return adapter.synthesize_to_audio_chunks(txt)

        def _on_metrics(m: dict) -> None:
            merged = dict(m or {})
            if voice:
                merged.setdefault("engine", "clone")
                merged.setdefault("clone_engine", clone_engine_name)
                merged.setdefault("voice_id", str(voice))
                merged.setdefault("speed", float(clone_speed))
            else:
                merged.setdefault("engine", engine_id)
                merged.setdefault("speed", float(getattr(self, "speed", 1.0) or 1.0))
            merged.setdefault("language", str(getattr(self, "language", None) or "en"))
            try:
                self._set_last_tts_metrics(merged)
            except Exception:
                pass

        stream = TextToSpeechStream(
            iter_audio_chunks_for_segment=_iter_chunks,
            on_audio_chunk=_on_audio_chunk,
            cancel_event=cancel,
            is_paused=_is_paused,
            on_metrics=_on_metrics,
        ).start()
        return stream

    def speak_to_bytes(
        self,
        text: str,
        format: str = "wav",
        voice: str | None = None,
        *,
        instructions: str | None = None,
        sanitize_syntax: bool = True,
        saninitze_syntax: bool | None = None,
    ) -> bytes:
        """Synthesize to bytes.

        - If `voice` is None: use the active TTS engine/adapter (default: Piper).
        - If `voice` is provided: treat as a cloned voice_id (requires a cloning
          backend extra such as `abstractvoice[omnivoice]`; `abstractvoice[cloning]`
          is the explicit OpenF5 backend).
        """
        # Clear prior metrics for this new utterance.
        self._set_last_tts_metrics(None)

        fmt = str(format or "wav").strip().lower() or "wav"
        request = self._build_speech_request(
            str(text),
            voice=voice,
            instructions=instructions,
            output_format=fmt,
            sanitize_syntax=_resolve_sanitize_syntax_arg(sanitize_syntax, saninitze_syntax),
            speed=float(getattr(self, "speed", 1.0) or 1.0),
        )
        speak_text = str(request.text)

        def _analyze_audio_bytes(b: bytes) -> dict:
            metrics: dict = {}
            try:
                import io

                import soundfile as sf

                info = sf.info(io.BytesIO(bytes(b)))
                try:
                    metrics["sample_rate"] = int(getattr(info, "samplerate", 0) or 0) or None
                except Exception:
                    metrics["sample_rate"] = None
                try:
                    metrics["channels"] = int(getattr(info, "channels", 0) or 0) or None
                except Exception:
                    metrics["channels"] = None
                try:
                    frames = int(getattr(info, "frames", 0) or 0)
                    metrics["audio_frames"] = frames if frames > 0 else None
                except Exception:
                    metrics["audio_frames"] = None
                try:
                    d = float(getattr(info, "duration", 0.0) or 0.0)
                    metrics["audio_s"] = float(d) if d > 0 else None
                except Exception:
                    metrics["audio_s"] = None
            except Exception:
                pass
            return metrics

        t0 = time.monotonic()
        if voice:
            cloner = self._get_voice_cloner()
            clone_speed, clone_engine_name = self._effective_clone_speed(
                cloner=cloner,
                voice_id=str(voice),
                surface="speak_bytes",
            )
            out = cloner.speak_to_bytes(
                speak_text,
                voice_id=str(request.voice or voice),
                format=fmt,
                speed=clone_speed,
                language=str(getattr(self, "language", None) or "en"),
            )
            synth_s = float(time.monotonic() - t0)

            metrics = {
                "engine": "clone",
                "clone_engine": clone_engine_name,
                "voice_id": str(request.voice or voice),
                "streaming": False,
                "synth_s": synth_s,
                "format": fmt,
                "speed": float(clone_speed),
                "language": str(getattr(self, "language", None) or "en"),
                "request_contract": "speech_request_v1",
                "ts": time.time(),
            }
            metrics.update(_analyze_audio_bytes(bytes(out)))
            try:
                audio_s = metrics.get("audio_s")
                if isinstance(audio_s, (int, float)) and float(audio_s) > 0:
                    metrics["rtf"] = float(synth_s) / float(audio_s)
            except Exception:
                pass
            self._set_last_tts_metrics(metrics)
            return out

        if self.tts_adapter and self.tts_adapter.is_available():
            provider_name = str(
                getattr(self.tts_adapter, "engine_id", None)
                or getattr(self.tts_adapter, "provider", None)
                or getattr(self, "_tts_engine_name", None)
                or getattr(self, "_tts_engine_preference", None)
                or ""
            ).strip().lower() or None
            model_name = str(
                getattr(self.tts_adapter, "model_id", None)
                or getattr(self, "tts_model", None)
                or ""
            ).strip() or None
            effective_voice = str(request.voice or "").strip() or None
            effective_speed = float(request.speed) if request.speed is not None else None
            effective_instructions = str(request.instructions) if request.instructions is not None else None
            if effective_voice and self._compatibility_support_level(
                kind="tts",
                provider=provider_name,
                model=model_name,
                surface="bytes",
                feature="profile",
            ) == "unsupported":
                effective_voice = None
            if effective_instructions is not None and self._compatibility_support_level(
                kind="tts",
                provider=provider_name,
                model=model_name,
                surface="bytes",
                feature="instructions",
            ) == "unsupported":
                effective_instructions = None
            if effective_speed is not None and self._compatibility_support_level(
                kind="tts",
                provider=provider_name,
                model=model_name,
                surface="bytes",
                feature="speed",
            ) == "unsupported":
                effective_speed = None
            use_voice_aware_bytes = bool(
                hasattr(self.tts_adapter, "synthesize_to_bytes_with_voice")
                and (
                    effective_instructions
                    or effective_voice
                    or (
                        isinstance(effective_speed, (int, float))
                        and float(effective_speed) != 1.0
                    )
                )
            )
            if use_voice_aware_bytes:
                out = self.tts_adapter.synthesize_to_bytes_with_voice(
                    speak_text,
                    format=fmt,
                    voice=effective_voice,
                    speed=effective_speed,
                    instructions=effective_instructions,
                )
            else:
                out = self.tts_adapter.synthesize_to_bytes(speak_text, format=fmt)
            synth_s = float(time.monotonic() - t0)
            try:
                engine_id = str(getattr(self.tts_adapter, "engine_id", "") or "").strip().lower()
            except Exception:
                engine_id = ""
            metrics = {
                "engine": engine_id or "tts",
                "synth_s": synth_s,
                "format": fmt,
                "language": str(getattr(self, "language", None) or "en"),
                "request_contract": "speech_request_v1",
                "ts": time.time(),
            }
            # Best-effort: attach active profile info when supported by the adapter.
            try:
                p = getattr(self.tts_adapter, "get_active_profile", None)
                prof = p() if callable(p) else None
                if prof is not None:
                    metrics["profile_id"] = getattr(prof, "profile_id", None)
                    metrics["profile_label"] = getattr(prof, "label", None)
            except Exception:
                pass
            metrics.update(_analyze_audio_bytes(bytes(out)))
            try:
                audio_s = metrics.get("audio_s")
                if isinstance(audio_s, (int, float)) and float(audio_s) > 0:
                    metrics["rtf"] = float(synth_s) / float(audio_s)
            except Exception:
                pass
            self._set_last_tts_metrics(metrics)
            return out
        raise RuntimeError(self._tts_adapter_unavailable_message(getattr(self, "tts_adapter", None)))

    def speak_to_file(
        self,
        text: str,
        output_path: str,
        format: str | None = None,
        voice: str | None = None,
        *,
        sanitize_syntax: bool = True,
        saninitze_syntax: bool | None = None,
    ) -> str:
        # Clear prior metrics for this new utterance.
        self._set_last_tts_metrics(None)

        sanitize = _resolve_sanitize_syntax_arg(sanitize_syntax, saninitze_syntax)
        fmt_hint = str(format or "").strip().lower() or None
        request = self._build_speech_request(
            str(text),
            voice=voice,
            output_format=fmt_hint,
            sanitize_syntax=sanitize,
            speed=float(getattr(self, "speed", 1.0) or 1.0),
        )
        speak_text = str(request.text)
        if voice:
            from pathlib import Path

            # For cloned voices, we only have a bytes API; write it out here.
            fmt = str(format or Path(output_path).suffix.lstrip(".") or "wav").strip().lower() or "wav"
            data = self.speak_to_bytes(speak_text, format=fmt, voice=voice, sanitize_syntax=False)
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(bytes(data))
            return str(out)

        if self.tts_adapter and self.tts_adapter.is_available():
            t0 = time.monotonic()
            out_path = self.tts_adapter.synthesize_to_file(speak_text, output_path, format=format)
            synth_s = float(time.monotonic() - t0)

            try:
                engine_id = str(getattr(self.tts_adapter, "engine_id", "") or "").strip().lower()
            except Exception:
                engine_id = ""

            fmt_used = None
            try:
                from pathlib import Path

                fmt_used = str(format or Path(str(out_path)).suffix.lstrip(".") or "wav").strip().lower() or "wav"
            except Exception:
                fmt_used = str(format or "wav").strip().lower() or "wav"

            metrics: dict = {
                "engine": engine_id or "tts",
                "synth_s": synth_s,
                "format": fmt_used,
                "language": str(getattr(self, "language", None) or "en"),
                "ts": time.time(),
            }
            # Best-effort: attach active profile info when supported by the adapter.
            try:
                p = getattr(self.tts_adapter, "get_active_profile", None)
                prof = p() if callable(p) else None
                if prof is not None:
                    metrics["profile_id"] = getattr(prof, "profile_id", None)
                    metrics["profile_label"] = getattr(prof, "label", None)
            except Exception:
                pass

            try:
                import soundfile as sf

                info = sf.info(str(out_path))
                try:
                    metrics["sample_rate"] = int(getattr(info, "samplerate", 0) or 0) or None
                except Exception:
                    metrics["sample_rate"] = None
                try:
                    metrics["channels"] = int(getattr(info, "channels", 0) or 0) or None
                except Exception:
                    metrics["channels"] = None
                try:
                    frames = int(getattr(info, "frames", 0) or 0)
                    metrics["audio_frames"] = frames if frames > 0 else None
                except Exception:
                    metrics["audio_frames"] = None
                try:
                    d = float(getattr(info, "duration", 0.0) or 0.0)
                    metrics["audio_s"] = float(d) if d > 0 else None
                except Exception:
                    metrics["audio_s"] = None
            except Exception:
                pass

            try:
                audio_s = metrics.get("audio_s")
                if isinstance(audio_s, (int, float)) and float(audio_s) > 0:
                    metrics["rtf"] = float(synth_s) / float(audio_s)
            except Exception:
                pass

            self._set_last_tts_metrics(metrics)
            return str(out_path)

        raise RuntimeError(self._tts_adapter_unavailable_message(getattr(self, "tts_adapter", None)))

    def stop_speaking(self):
        if not self.tts_engine:
            return False
        # Best-effort cancel ongoing cloned synthesis.
        try:
            cancel = getattr(self, "_cloned_cancel_event", None)
            if cancel is not None:
                cancel.set()
        except Exception:
            pass
        ok = False
        try:
            # Keep the output stream open when possible; repeatedly reopening
            # PortAudio streams can be flaky on some macOS AUHAL setups.
            try:
                ok = bool(self.tts_engine.stop(close_stream=False))
            except TypeError:
                ok = bool(self.tts_engine.stop())
        finally:
            # CRITICAL: stopping playback abruptly may not trigger the normal
            # playback-end callbacks (PortAudio stream is just closed).
            # If we don't restore recognizer state here, transcriptions can stay
            # paused or listening can remain paused, which breaks STOP/PTT.
            try:
                on_end = getattr(self, "_on_tts_end", None)
                if callable(on_end):
                    on_end()
            except Exception:
                pass
        return ok

    def pause_speaking(self):
        if not self.tts_engine:
            return False
        return self.tts_engine.pause()

    def resume_speaking(self):
        if not self.tts_engine:
            return False
        return self.tts_engine.resume()

    def is_paused(self):
        if not self.tts_engine:
            return False
        return self.tts_engine.is_paused()

    def is_speaking(self):
        if self.tts_engine:
            return self.tts_engine.is_active()
        return False

    def set_speed(self, speed):
        try:
            sp = float(speed)
        except Exception:
            return False
        if not (0.5 <= sp <= 2.0):
            return False

        try:
            a = getattr(self, "tts_adapter", None)
            provider_name = str(
                getattr(a, "engine_id", None)
                or getattr(a, "provider", None)
                or getattr(self, "_tts_engine_name", None)
                or getattr(self, "_tts_engine_preference", None)
                or ""
            ).strip().lower() or None
            model_name = str(
                getattr(a, "model_id", None)
                or getattr(self, "tts_model", None)
                or ""
            ).strip() or None
        except Exception:
            provider_name = None
            model_name = None
        if sp != 1.0 and self._compatibility_support_level(
            kind="tts",
            provider=provider_name,
            model=model_name,
            surface="playback",
            feature="speed",
        ) == "unsupported":
            # Keep manager speed unchanged (or reset to 1.0 if unset).
            try:
                self.speed = float(getattr(self, "speed", 1.0) or 1.0)
            except Exception:
                self.speed = 1.0
            return False

        self.speed = float(sp)
        return True

    def get_speed(self):
        return self.speed

    def _try_init_piper(self, language: str):
        try:
            from ..adapters.tts_piper import PiperTTSAdapter
            adapter = PiperTTSAdapter(
                language=language,
                allow_downloads=bool(getattr(self, "allow_downloads", True)),
                auto_load=True,
            )
            # Return the adapter even if a voice is not yet loaded. This keeps audio
            # playback available for cloning backends while remaining offline-first.
            return adapter if bool(getattr(adapter, "_piper_available", False)) else None
        except Exception as e:
            if self.debug_mode:
                print(f"⚠️  Piper TTS not available: {e}")
            return None

    def get_supported_languages(self):
        adapter = getattr(self, "tts_adapter", None)
        if adapter is not None and hasattr(adapter, "get_supported_languages"):
            try:
                langs = list(adapter.get_supported_languages() or [])
                if langs:
                    return langs
            except Exception:
                pass
        return list(self.LANGUAGES.keys())

    def list_available_models(self, language: str | None = None) -> dict:
        """List available TTS voices/models for the active adapter.

        Returns a dict shaped for CLI display:
        { "en": { "amy": { ... } }, "fr": { ... } }
        """
        if self.tts_adapter and hasattr(self.tts_adapter, "list_available_models"):
            return self.tts_adapter.list_available_models(language=language)

        # Best-effort: instantiate a temporary Piper adapter to enumerate models.
        try:
            from ..adapters.tts_piper import PiperTTSAdapter

            return PiperTTSAdapter(
                language=(language or "en"),
                allow_downloads=False,
                auto_load=False,
            ).list_available_models(language=language)
        except Exception:
            return {}

    # Backward-compatible alias used by some CLI code.
    def list_voices(self, language: str | None = None) -> dict:
        return self.list_available_models(language=language)

    def get_language(self):
        return self.language

    def get_language_name(self, language_code=None):
        lang = language_code or self.language
        return self.LANGUAGES.get(lang, {}).get("name", lang)

    def set_language(self, language):
        language = str(language or "").strip().lower()
        if not language:
            return False

        # Language validation is engine-dependent:
        # - Piper uses a small curated mapping to avoid trying to
        #   load non-existent voices.
        # - Other engines (e.g. OmniVoice) can support many languages; treat the
        #   language code as a pass-through hint and let the engine decide.
        pref = str(getattr(self, "_tts_engine_preference", "auto") or "auto").strip().lower()
        active_engine = ""
        try:
            a = getattr(self, "tts_adapter", None)
            active_engine = str(getattr(a, "engine_id", "") or "").strip().lower()
        except Exception:
            active_engine = ""

        if active_engine:
            validate_against_catalog = active_engine == "piper"
        else:
            validate_against_catalog = pref in ("piper",)
        if validate_against_catalog:
            supported = set()
            try:
                if self.tts_adapter is not None:
                    supported = set(self.tts_adapter.get_supported_languages())
            except Exception:
                supported = set()
            if not supported:
                try:
                    from ..adapters.tts_piper import PiperTTSAdapter

                    supported = set(PiperTTSAdapter.PIPER_MODELS.keys())
                except Exception:
                    supported = {"en", "fr", "de", "es", "ru", "zh"}
        else:
            supported = set()
        if validate_against_catalog and language not in supported:
            if self.debug_mode:
                available = ", ".join(sorted(supported))
                print(f"⚠️ Unsupported language '{language}'. Available: {available}")
            return False

        if language == self.language:
            if self.debug_mode:
                print(f"✓ Already using {self.get_language_name(language)} voice")
            return True

        self.stop_speaking()
        if self.voice_recognizer:
            self.voice_recognizer.stop()
            self.voice_recognizer = None

        # Switch language on the active TTS adapter (engine-agnostic).
        try:
            if self.tts_adapter is None:
                pref = str(getattr(self, "_tts_engine_preference", "auto") or "auto")
                self.tts_adapter, resolved_engine = create_tts_adapter(
                    engine=pref,
                    language=language,
                    allow_downloads=bool(getattr(self, "allow_downloads", True)),
                    auto_load=False,
                    debug_mode=bool(getattr(self, "debug_mode", False)),
                    model_id=getattr(self, "tts_model", None),
                    base_url=getattr(self, "remote_base_url", None),
                    api_key=getattr(self, "remote_api_key", None),
                    timeout_s=getattr(self, "remote_timeout_s", None),
                )
                if self.tts_adapter is None:
                    return False
                # Track which engine is active (used by CLI/tests/metrics).
                self._tts_engine_name = str(resolved_engine)
                if self.tts_engine is None:
                    from ..tts.adapter_tts_engine import AdapterTTSEngine

                    self.tts_engine = AdapterTTSEngine(self.tts_adapter, debug_mode=self.debug_mode)
                    self._wire_tts_callbacks()

            ok = bool(self.tts_adapter.set_language(language))
            if not ok:
                return False

            # Ensure playback wrapper exists (used for lifecycle callbacks + audio output).
            if self.tts_engine is None and self.tts_adapter is not None:
                from ..tts.adapter_tts_engine import AdapterTTSEngine

                self.tts_engine = AdapterTTSEngine(self.tts_adapter, debug_mode=self.debug_mode)
                if not getattr(self, "_tts_engine_name", None):
                    pref = str(getattr(self, "_tts_engine_preference", "auto") or "auto").strip().lower()
                    self._tts_engine_name = "piper" if pref in ("", "auto") else pref
                self._wire_tts_callbacks()

            self.language = language
            self.speed = 1.0
            self.reset_tts_profile(language=language)
            return True
        except Exception as e:
            if self.debug_mode:
                print(f"⚠️ TTS language switch failed: {e}")

        return False

    def set_voice(self, language, voice_id):
        language = language.lower()

        # Piper voice selection is adapter-specific. For now, treat `voice_id` as
        # best-effort metadata and ensure language switching is robust.
        try:
            if not self.set_language(language):
                return False
            if self.debug_mode:
                print(f"🎭 Piper voice selection (best-effort): {language}.{voice_id}")
            return True
        except Exception:
            return False
