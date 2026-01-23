"""Adapter-backed TTSEngine facade.

Why this exists
---------------
`VoiceManager` historically relied on a `TTSEngine` instance exposing:
- speak(text, speed, callback)
- stop/pause/resume/is_active/is_paused
- on_playback_start/on_playback_end callbacks
- an `audio_player` that supports immediate pause/resume

With the introduction of adapter-based engines (e.g. Piper), `VoiceManager`
must keep that internal contract stable to preserve backward compatibility
across the codebase (CLI, tests, integrations).

This module provides a small engine facade that wraps any `TTSAdapter` and
uses the existing `NonBlockingAudioPlayer` for playback control.
"""

from __future__ import annotations

import threading
from typing import Callable, Optional

import numpy as np

from .tts_engine import NonBlockingAudioPlayer, apply_speed_without_pitch_change
from ..adapters.base import TTSAdapter


class AdapterTTSEngine:
    """TTSEngine-compatible wrapper around a `TTSAdapter`."""

    def __init__(self, adapter: TTSAdapter, debug_mode: bool = False):
        self.adapter = adapter
        self.debug_mode = debug_mode

        self.on_playback_start: Optional[Callable[[], None]] = None
        self.on_playback_end: Optional[Callable[[], None]] = None

        self._user_callback: Optional[Callable[[], None]] = None

        sample_rate = self._safe_sample_rate()
        self.audio_player = NonBlockingAudioPlayer(sample_rate=sample_rate, debug_mode=debug_mode)
        self.audio_player.playback_complete_callback = self._on_playback_complete

    def _safe_sample_rate(self) -> int:
        try:
            return int(self.adapter.get_sample_rate())
        except Exception:
            return 22050

    def _sync_sample_rate(self) -> None:
        """Keep audio player sample rate aligned with adapter."""
        sr = self._safe_sample_rate()
        if getattr(self.audio_player, "sample_rate", None) == sr:
            return

        # Changing sample rate requires restarting the underlying stream.
        was_playing = bool(getattr(self.audio_player, "is_playing", False))
        try:
            self.audio_player.stop_stream()
        except Exception:
            # Best-effort; don't crash on cleanup issues.
            pass

        self.audio_player.sample_rate = sr

        # If we were mid-playback we cannot safely resume; callers can re-speak.
        if was_playing and self.debug_mode:
            print(f"⚠️  Audio sample rate changed to {sr}Hz; playback was stopped")

    def speak(self, text: str, speed: float = 1.0, callback=None) -> bool:
        """Synthesize and enqueue audio for playback (non-blocking)."""
        if not self.adapter or not self.adapter.is_available():
            raise RuntimeError("No TTS adapter available")

        self._sync_sample_rate()

        self._user_callback = callback

        if self.on_playback_start:
            threading.Thread(target=self.on_playback_start, daemon=True).start()

        audio: np.ndarray = self.adapter.synthesize(text)

        # Best-effort speed handling. If librosa isn't installed, the helper
        # falls back to original audio (no crash).
        if speed and speed != 1.0:
            audio = apply_speed_without_pitch_change(audio, speed, sr=self._safe_sample_rate())

        self.audio_player.play_audio(audio)
        return True

    def begin_playback(self, callback=None, *, sample_rate: int | None = None) -> None:
        """Begin a playback session without synthesizing.

        Used for streaming/chunked playback where audio is enqueued progressively.
        """
        if sample_rate is not None:
            # For streaming audio produced externally (e.g. cloning), prefer
            # playing at the native sample rate to avoid resampling artifacts.
            sr = int(sample_rate)
            if getattr(self.audio_player, "sample_rate", None) != sr:
                try:
                    self.audio_player.stop_stream()
                except Exception:
                    pass
                self.audio_player.sample_rate = sr
        else:
            self._sync_sample_rate()
        if callback is not None:
            self._user_callback = callback
        if self.on_playback_start:
            threading.Thread(target=self.on_playback_start, daemon=True).start()

    def enqueue_audio(self, audio: np.ndarray) -> None:
        """Enqueue audio into the underlying player (no extra callbacks)."""
        self.audio_player.play_audio(audio)

    def play_audio_array(self, audio: np.ndarray, callback=None) -> bool:
        """Play already-synthesized audio through the same playback pipeline.

        Used for optional features (e.g., voice cloning) that produce WAV bytes
        externally but still want to reuse the existing low-latency playback +
        lifecycle callbacks.
        """
        self._user_callback = callback
        if self.on_playback_start:
            threading.Thread(target=self.on_playback_start, daemon=True).start()

        self.audio_player.play_audio(audio)
        return True

    def _on_playback_complete(self) -> None:
        """Called by the audio player when playback fully drains."""
        if self.on_playback_end:
            threading.Thread(target=self.on_playback_end, daemon=True).start()

        if self._user_callback:
            threading.Thread(target=self._user_callback, daemon=True).start()
            self._user_callback = None

    def stop(self) -> bool:
        """Stop playback immediately and clear queued audio."""
        # Even if `is_playing` is False, an OutputStream may still be alive.
        # Always stop the stream if it exists to avoid leaving PortAudio
        # resources around (which can crash on interpreter shutdown).
        stream_exists = getattr(self.audio_player, "stream", None) is not None
        was_playing = bool(getattr(self.audio_player, "is_playing", False))

        if not (stream_exists or was_playing):
            return False

        self.audio_player.stop_stream()
        return True

    def pause(self) -> bool:
        return self.audio_player.pause()

    def resume(self) -> bool:
        return self.audio_player.resume()

    def is_paused(self) -> bool:
        return self.audio_player.is_paused_state()

    def is_active(self) -> bool:
        return bool(getattr(self.audio_player, "is_playing", False))

    def cleanup(self) -> None:
        try:
            self.audio_player.cleanup()
        except Exception:
            pass

