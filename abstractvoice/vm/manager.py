"""Small public façade for VoiceManager.

The heavy implementation is split across focused mixins to keep files small
and responsibilities clear.
"""

from __future__ import annotations

from typing import Optional

from ..config.voice_catalog import LANGUAGES, SAFE_FALLBACK, VOICE_CATALOG
from ..tts.adapter_tts_engine import AdapterTTSEngine

from .common import import_tts_engine
from .core import VoiceManagerCore
from .mm_mixin import MMMixin
from .stt_mixin import SttMixin
from .tts_mixin import TtsMixin


class VoiceManager(VoiceManagerCore, TtsMixin, SttMixin, MMMixin):
    """Main class for voice interaction capabilities."""

    LANGUAGES = LANGUAGES
    SAFE_FALLBACK = SAFE_FALLBACK
    VOICE_CATALOG = VOICE_CATALOG

    def __init__(
        self,
        language: str = "en",
        tts_model: Optional[str] = None,
        whisper_model: str = "tiny",
        debug_mode: bool = False,
        tts_engine: str = "auto",
        stt_engine: str = "auto",
    ):
        self.debug_mode = debug_mode
        self.speed = 1.0

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

        if tts_engine in ("auto", "piper"):
            self.tts_adapter = self._try_init_piper(language)
            if self.tts_adapter and self.tts_adapter.is_available():
                self.tts_engine = AdapterTTSEngine(self.tts_adapter, debug_mode=debug_mode)
                self._tts_engine_name = "piper"

        if self.tts_engine is None and tts_engine in ("auto", "vits"):
            if tts_model is None:
                tts_model = self._select_best_model(self.language)
                if debug_mode:
                    lang_name = self.LANGUAGES[self.language]["name"]
                    print(f"🌍 Using {lang_name} voice: {tts_model}")

            # Legacy TTSEngine
            TTSEngine = import_tts_engine()
            self.tts_engine = TTSEngine(model_name=tts_model, debug_mode=debug_mode)
            self._tts_engine_name = "vits"

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

        # State tracking
        self._transcription_callback = None
        self._stop_callback = None
        self._voice_mode = "full"

