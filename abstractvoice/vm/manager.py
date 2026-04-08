"""Small public façade for VoiceManager.

The heavy implementation is split across focused mixins to keep files small
and responsibilities clear.
"""

from __future__ import annotations

import threading
from typing import Optional

from ..config.voice_catalog import LANGUAGES, SAFE_FALLBACK
from ..adapters.tts_registry import create_tts_adapter
from ..tts.adapter_tts_engine import AdapterTTSEngine
from ..voice_profiles import VoiceProfile

from .core import VoiceManagerCore
from .stt_mixin import SttMixin
from .tts_mixin import TtsMixin


class VoiceManager(VoiceManagerCore, TtsMixin, SttMixin):
    """Main class for voice interaction capabilities."""

    LANGUAGES = LANGUAGES
    SAFE_FALLBACK = SAFE_FALLBACK

    def __init__(
        self,
        language: str = "en",
        tts_model: Optional[str] = None,
        # Default STT model: "base" is a better out-of-box quality baseline than "tiny",
        # especially for short commands and non-ideal microphone conditions.
        whisper_model: str = "base",
        debug_mode: bool = False,
        tts_engine: str = "auto",
        stt_engine: str = "auto",
        allow_downloads: bool = True,
        cloned_tts_streaming: bool = True,
        cloning_engine: str = "f5_tts",
        tts_delivery_mode: str | None = None,
    ):
        self.debug_mode = debug_mode
        self.speed = 1.0
        # Controls whether the library may download model weights implicitly.
        # The REPL sets this to False to enforce "no surprise downloads".
        self.allow_downloads = bool(allow_downloads)
        # Cloned TTS can either stream batches (lower time-to-first-audio, but may
        # introduce gaps if generation can't stay ahead) or generate full audio first.
        self.cloned_tts_streaming = bool(cloned_tts_streaming)
        # Unified delivery-mode override (applies to base TTS and cloned voices).
        # When unset, base TTS uses buffered delivery and cloned voices use `cloned_tts_streaming`.
        self.tts_delivery_mode: str | None = None
        if tts_delivery_mode is not None and str(tts_delivery_mode).strip():
            from ..tts.delivery_mode import normalize_audio_delivery_mode

            self.tts_delivery_mode = normalize_audio_delivery_mode(tts_delivery_mode)
        self.cloning_engine = str(cloning_engine or "f5_tts").strip().lower()

        requested_engine = str(tts_engine or "auto").strip().lower() or "auto"

        # Language normalization:
        # - For Piper (default/auto), keep the historical catalog validation so
        #   we don't try to load non-existent voices.
        # - For other engines (e.g. OmniVoice), allow arbitrary language codes
        #   and let the engine decide (some engines support 100s of languages).
        language = str(language or "en").strip().lower() or "en"
        if requested_engine in ("", "auto", "piper"):
            if language not in self.LANGUAGES:
                if debug_mode:
                    available = ", ".join(self.LANGUAGES.keys())
                    print(f"⚠️ Unsupported language '{language}', using English. Available: {available}")
                language = "en"
        self.language = language

        self._tts_engine_preference = tts_engine
        self._stt_engine_preference = stt_engine

        # TTS selection
        self.tts_adapter = None
        self._tts_engine_name = None
        self.tts_engine = None

        # Create the playback engine as long as the selected adapter runtime is
        # importable. This keeps audio output available for cloning backends even
        # when no TTS model is cached locally (offline-first).
        try:
            self.tts_adapter, resolved_engine = create_tts_adapter(
                engine=str(tts_engine or "auto"),
                language=language,
                allow_downloads=bool(self.allow_downloads),
                auto_load=True,
                debug_mode=bool(debug_mode),
            )
        except ValueError:
            # Preserve caller-facing validation semantics (explicit engine names must be valid).
            raise
        except Exception as e:
            # If the caller explicitly selected an engine, surface the error so
            # it's actionable (no silent fallback).
            if requested_engine != "auto":
                raise
            if debug_mode:
                print(f"⚠️  TTS engine init failed: {e}")
            self.tts_adapter = None
            resolved_engine = None

        if self.tts_adapter:
            self.tts_engine = AdapterTTSEngine(self.tts_adapter, debug_mode=debug_mode)
            self._tts_engine_name = resolved_engine

        # Audio lifecycle callbacks (public hooks)
        self.on_audio_start = None
        self.on_audio_end = None
        self.on_audio_pause = None
        self.on_audio_resume = None

        self._wire_tts_callbacks()

        # STT / listening
        self.voice_recognizer = None
        self.whisper_model = whisper_model
        self.stt_adapter = None
        self._voice_cloner = None
        self._aec_enabled = False
        self._aec_stream_delay_ms = 0

        # Cloned-speech cancellation token (best-effort).
        self._cloned_cancel_event = threading.Event()

        # Tracks whether cloned TTS synthesis is currently running (separate from playback).
        self._cloned_synthesis_active = threading.Event()

        # Best-effort last TTS metrics (used by verbose REPL output).
        self._last_tts_metrics = None
        self._last_tts_metrics_lock = threading.Lock()

        # State tracking
        self._transcription_callback = None
        self._stop_callback = None
        # Default to "wait" for robustness without echo cancellation.
        # "full" is intended for headset / echo-controlled environments.
        self._voice_mode = "wait"

    # ------------------------------------------------------------------
    # Voice profiles (cross-engine)
    # ------------------------------------------------------------------

    def get_profiles(self, *, kind: str = "tts") -> list[VoiceProfile]:
        """List available profiles for the active engine (best-effort)."""
        k = str(kind or "").strip().lower() or "tts"
        if k != "tts":
            raise ValueError("Only kind='tts' is supported for now.")
        adapter = getattr(self, "tts_adapter", None)
        if adapter is None:
            return []
        try:
            out = adapter.get_profiles()
            return list(out) if isinstance(out, list) else list(out or [])
        except Exception:
            return []

    def set_profile(self, profile_id: str, *, kind: str = "tts") -> bool:
        """Apply a profile by id for the active engine."""
        k = str(kind or "").strip().lower() or "tts"
        if k != "tts":
            raise ValueError("Only kind='tts' is supported for now.")
        adapter = getattr(self, "tts_adapter", None)
        if adapter is None:
            return False
        return bool(adapter.set_profile(str(profile_id)))

    def get_active_profile(self, *, kind: str = "tts") -> VoiceProfile | None:
        """Return the active profile for the active engine (best-effort)."""
        k = str(kind or "").strip().lower() or "tts"
        if k != "tts":
            raise ValueError("Only kind='tts' is supported for now.")
        adapter = getattr(self, "tts_adapter", None)
        if adapter is None:
            return None
        try:
            p = adapter.get_active_profile()
            return p if isinstance(p, VoiceProfile) else None
        except Exception:
            return None
