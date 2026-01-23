"""VoiceManager core (init + lifecycle callbacks + cleanup)."""

from __future__ import annotations


class VoiceManagerCore:
    """Core orchestration (shared state and callbacks)."""

    def _wire_tts_callbacks(self) -> None:
        if self.tts_engine is None:
            return

        # TTS lifecycle used to coordinate listening modes.
        self.tts_engine.on_playback_start = self._on_tts_start
        self.tts_engine.on_playback_end = self._on_tts_end

        # Audio lifecycle callbacks (actual playback).
        if hasattr(self.tts_engine, "audio_player") and self.tts_engine.audio_player:
            self.tts_engine.audio_player.on_audio_start = self._on_audio_start
            self.tts_engine.audio_player.on_audio_end = self._on_audio_end
            self.tts_engine.audio_player.on_audio_pause = self._on_audio_pause
            self.tts_engine.audio_player.on_audio_resume = self._on_audio_resume

    def _on_tts_start(self):
        """Called when TTS playback starts - handle based on voice mode."""
        if not self.voice_recognizer:
            return

        if self._voice_mode == "full":
            # Full mode: keep listening but prevent self-interrupt.
            self.voice_recognizer.pause_tts_interrupt()
        elif self._voice_mode in ["wait", "stop", "ptt"]:
            # Wait/Stop/PTT: pause listening entirely during TTS.
            self.voice_recognizer.pause_listening()

    def _on_tts_end(self):
        """Called when TTS playback ends - handle based on voice mode."""
        if not self.voice_recognizer:
            return

        if self._voice_mode == "full":
            self.voice_recognizer.resume_tts_interrupt()
        elif self._voice_mode in ["wait", "stop", "ptt"]:
            self.voice_recognizer.resume_listening()

    def _on_audio_start(self):
        if self.on_audio_start:
            self.on_audio_start()

    def _on_audio_end(self):
        if self.on_audio_end:
            self.on_audio_end()

    def _on_audio_pause(self):
        if self.on_audio_pause:
            self.on_audio_pause()

    def _on_audio_resume(self):
        if self.on_audio_resume:
            self.on_audio_resume()

    def cleanup(self):
        """Clean up resources."""
        if self.voice_recognizer:
            self.voice_recognizer.stop()

        self.stop_speaking()

        # Best-effort: fully release audio resources.
        try:
            if self.tts_engine is not None:
                if hasattr(self.tts_engine, "cleanup"):
                    self.tts_engine.cleanup()
                elif hasattr(self.tts_engine, "audio_player") and self.tts_engine.audio_player:
                    self.tts_engine.audio_player.cleanup()
        except Exception:
            pass

        return True

