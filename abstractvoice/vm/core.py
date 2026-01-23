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
            # Optional: feed far-end playback audio to the listener for AEC.
            try:
                self.tts_engine.audio_player.on_audio_chunk = self._on_audio_chunk
            except Exception:
                pass

    def _on_audio_chunk(self, audio_chunk, sample_rate: int):
        """Called with chunks actually written to speaker output.

        This is used only for advanced features like AEC-based barge-in.
        """
        if not self.voice_recognizer:
            return
        if hasattr(self.voice_recognizer, "feed_far_end_audio"):
            try:
                self.voice_recognizer.feed_far_end_audio(audio_chunk, sample_rate=sample_rate)
            except Exception:
                pass

    def _on_tts_start(self):
        """Called when TTS playback starts - handle based on voice mode."""
        if not self.voice_recognizer:
            return

        if self._voice_mode == "full":
            # Full mode: allow real barge-in ONLY if AEC is enabled.
            # Without AEC, pausing TTS interrupt avoids self-interruption.
            if not bool(getattr(self.voice_recognizer, "aec_enabled", False)):
                self.voice_recognizer.pause_tts_interrupt()
            # In full mode we keep transcriptions enabled for headset/echo-controlled setups.
        elif self._voice_mode in ["wait", "stop", "ptt"]:
            # Wait/Stop/PTT: disable generic interrupt and suppress normal transcriptions
            # to avoid self-feedback loops. Stop phrase remains available.
            self.voice_recognizer.pause_tts_interrupt()
            if hasattr(self.voice_recognizer, "pause_transcriptions"):
                self.voice_recognizer.pause_transcriptions()

    def _on_tts_end(self):
        """Called when TTS playback ends - handle based on voice mode."""
        if not self.voice_recognizer:
            return

        if self._voice_mode == "full":
            self.voice_recognizer.resume_tts_interrupt()
        elif self._voice_mode in ["wait", "stop", "ptt"]:
            self.voice_recognizer.resume_tts_interrupt()
            if hasattr(self.voice_recognizer, "resume_transcriptions"):
                self.voice_recognizer.resume_transcriptions()

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

