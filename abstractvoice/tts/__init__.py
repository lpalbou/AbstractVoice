"""TTS utilities (Piper-first)."""

from .tts_engine import NonBlockingAudioPlayer, apply_speed_without_pitch_change

__all__ = ["NonBlockingAudioPlayer", "apply_speed_without_pitch_change"]