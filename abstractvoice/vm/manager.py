"""Small public façade for VoiceManager.

The heavy implementation is split across focused mixins to keep files small
and responsibilities clear.
"""

from __future__ import annotations

import threading
from typing import Optional

from ..config.voice_catalog import LANGUAGES, SAFE_FALLBACK
from ..tts.adapter_tts_engine import AdapterTTSEngine

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
        whisper_model: str = "tiny",
        debug_mode: bool = False,
        tts_engine: str = "auto",
        stt_engine: str = "auto",
        allow_downloads: bool = True,
        cloned_tts_streaming: bool = True,
    ):
        self.debug_mode = debug_mode
        self.speed = 1.0
        # Controls whether the library may download model weights implicitly.
        # The REPL sets this to False to enforce "no surprise downloads".
        self.allow_downloads = bool(allow_downloads)
        # Cloned TTS can either stream batches (lower time-to-first-audio, but may
        # introduce gaps if generation can't stay ahead) or generate full audio first.
        self.cloned_tts_streaming = bool(cloned_tts_streaming)

        language = (language or "en").lower()
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

        if tts_engine not in ("auto", "piper"):
            raise ValueError("Only Piper TTS is supported in AbstractVoice core. Use tts_engine='piper'.")

        if tts_engine in ("auto", "piper"):
            self.tts_adapter = self._try_init_piper(language)
            if self.tts_adapter and self.tts_adapter.is_available():
                self.tts_engine = AdapterTTSEngine(self.tts_adapter, debug_mode=debug_mode)
                self._tts_engine_name = "piper"

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

        # State tracking
        self._transcription_callback = None
        self._stop_callback = None
        # Default to "wait" for robustness without echo cancellation.
        # "full" is intended for headset / echo-controlled environments.
        self._voice_mode = "wait"

