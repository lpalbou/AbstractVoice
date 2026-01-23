"""STT + listening methods for VoiceManager."""

from __future__ import annotations

from typing import Optional

from .common import import_voice_recognizer


class SttMixin:
    def transcribe_from_bytes(self, audio_bytes: bytes, language: Optional[str] = None) -> str:
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name

        try:
            return self.transcribe_file(tmp_path, language=language)
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    def transcribe_file(self, audio_path: str, language: Optional[str] = None) -> str:
        stt = self._get_stt_adapter()
        if stt is not None:
            return stt.transcribe(audio_path, language=language)

        # Optional fallback to legacy Transcriber if present.
        from ..stt import Transcriber

        transcriber = Transcriber(model_name=self.whisper_model, debug_mode=self.debug_mode)
        result = transcriber.transcribe(audio_path)
        return result["text"] if result and "text" in result else ""

    def _get_stt_adapter(self):
        if self.stt_adapter is not None:
            return self.stt_adapter if self.stt_adapter.is_available() else None

        if self._stt_engine_preference not in ("auto", "faster_whisper"):
            return None

        try:
            from ..adapters.stt_faster_whisper import FasterWhisperAdapter

            self.stt_adapter = FasterWhisperAdapter(model_size=self.whisper_model, device="cpu", compute_type="int8")
            if self.stt_adapter.is_available():
                return self.stt_adapter
            return None
        except Exception as e:
            if self.debug_mode:
                print(f"⚠️  Faster-Whisper STT not available: {e}")
            self.stt_adapter = None
            return None

    def set_whisper(self, model_name):
        self.whisper_model = model_name
        if self.voice_recognizer:
            return self.voice_recognizer.change_whisper_model(model_name)

    def get_whisper(self):
        return self.whisper_model

    def listen(self, on_transcription, on_stop=None):
        self._transcription_callback = on_transcription
        self._stop_callback = on_stop

        if not self.voice_recognizer:
            def _transcription_handler(text):
                if self._transcription_callback:
                    self._transcription_callback(text)

            def _stop_handler():
                self.stop_listening()
                if self._stop_callback:
                    self._stop_callback()

            VoiceRecognizer = import_voice_recognizer()
            self.voice_recognizer = VoiceRecognizer(
                transcription_callback=_transcription_handler,
                stop_callback=_stop_handler,
                whisper_model=self.whisper_model,
                debug_mode=self.debug_mode,
            )

        return self.voice_recognizer.start(tts_interrupt_callback=self.stop_speaking)

    def stop_listening(self):
        if self.voice_recognizer:
            return self.voice_recognizer.stop()
        return False

    def pause_listening(self) -> bool:
        if self.voice_recognizer:
            self.voice_recognizer.pause_listening()
            return True
        return False

    def resume_listening(self) -> bool:
        if self.voice_recognizer:
            self.voice_recognizer.resume_listening()
            return True
        return False

    def is_listening(self):
        return self.voice_recognizer and self.voice_recognizer.is_running

    def set_voice_mode(self, mode):
        if mode in ["full", "wait", "stop", "ptt"]:
            self._voice_mode = mode
            return True
        return False

    def change_vad_aggressiveness(self, aggressiveness):
        if self.voice_recognizer:
            return self.voice_recognizer.change_vad_aggressiveness(aggressiveness)
        return False

