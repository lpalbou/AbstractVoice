"""Core audio playback utilities (Piper-first).

AbstractVoice core intentionally avoids shipping legacy Coqui-based TTSEngine
logic. This module contains only reusable audio utilities:
- `NonBlockingAudioPlayer` for low-latency pause/resume/stop
- `apply_speed_without_pitch_change` (optional librosa)
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Callable, Optional

import numpy as np


def _import_sounddevice():
    try:
        import sounddevice as sd
        return sd
    except ImportError as e:
        raise ImportError(
            "Audio playback requires sounddevice. Install with:\n"
            "  pip install abstractvoice\n"
            f"Original error: {e}"
        ) from e


def _import_librosa():
    try:
        import librosa
        return librosa
    except ImportError as e:
        raise ImportError(
            "Speed/pitch processing requires librosa. Install with:\n"
            "  pip install \"abstractvoice[audio-fx]\"\n"
            f"Original error: {e}"
        ) from e


def apply_speed_without_pitch_change(audio: np.ndarray, speed: float, sr: int = 22050) -> np.ndarray:
    """Apply speed change without affecting pitch (best-effort).

    If librosa is not installed (or fails), returns the original audio.
    """
    if not speed or speed == 1.0:
        return audio

    try:
        librosa = _import_librosa()
        return librosa.effects.time_stretch(audio, rate=float(speed))
    except Exception as e:
        logging.warning(f"Time-stretching failed: {e}, using original audio")
        return audio


class NonBlockingAudioPlayer:
    """Non-blocking audio player using OutputStream callbacks for pause/resume."""

    def __init__(self, sample_rate: int = 22050, debug_mode: bool = False):
        self.sample_rate = int(sample_rate)
        self.debug_mode = debug_mode

        self.audio_queue: "queue.Queue[np.ndarray]" = queue.Queue()
        self.stream = None
        self.is_playing = False

        self._pause_lock = threading.Lock()
        self._paused = False

        self.current_audio: Optional[np.ndarray] = None
        self.current_position = 0

        self.playback_complete_callback: Optional[Callable[[], None]] = None

        # Audio lifecycle callbacks
        self.on_audio_start: Optional[Callable[[], None]] = None
        self.on_audio_end: Optional[Callable[[], None]] = None
        self.on_audio_pause: Optional[Callable[[], None]] = None
        self.on_audio_resume: Optional[Callable[[], None]] = None
        self._audio_started = False

        # Optional hook: called with chunks that are actually written to the output.
        # Used for advanced features like AEC (barge-in without self-interruption).
        self.on_audio_chunk = None  # Callable[[np.ndarray, int], None] | None

    def _audio_callback(self, outdata, frames, _time, status):
        if status and self.debug_mode:
            print(f"Audio callback status: {status}")

        with self._pause_lock:
            if self._paused:
                outdata.fill(0)
                return

        try:
            if self.current_audio is None or self.current_position >= len(self.current_audio):
                try:
                    self.current_audio = self.audio_queue.get_nowait()
                    self.current_position = 0
                except queue.Empty:
                    outdata.fill(0)
                    if self.is_playing:
                        self.is_playing = False
                        self._audio_started = False
                        if self.on_audio_end:
                            threading.Thread(target=self.on_audio_end, daemon=True).start()
                        if self.playback_complete_callback:
                            threading.Thread(target=self.playback_complete_callback, daemon=True).start()
                    return

            remaining = len(self.current_audio) - self.current_position
            frames_to_output = min(frames, remaining)

            if frames_to_output > 0 and not self._audio_started:
                self._audio_started = True
                if self.on_audio_start:
                    threading.Thread(target=self.on_audio_start, daemon=True).start()

            if frames_to_output > 0:
                if outdata.shape[1] == 1:
                    outdata[:frames_to_output, 0] = self.current_audio[
                        self.current_position : self.current_position + frames_to_output
                    ]
                else:
                    audio_data = self.current_audio[
                        self.current_position : self.current_position + frames_to_output
                    ]
                    outdata[:frames_to_output, 0] = audio_data
                    outdata[:frames_to_output, 1] = audio_data

                # Emit the actual output chunk (mono float32) for optional consumers.
                try:
                    if self.on_audio_chunk:
                        chunk = self.current_audio[
                            self.current_position : self.current_position + frames_to_output
                        ]
                        self.on_audio_chunk(chunk, int(self.sample_rate))
                except Exception:
                    # Never let optional hooks break audio playback.
                    pass
                self.current_position += frames_to_output

            if frames_to_output < frames:
                outdata[frames_to_output:].fill(0)

        except Exception as e:
            if self.debug_mode:
                print(f"Error in audio callback: {e}")
            outdata.fill(0)

    def start_stream(self):
        if self.stream is not None:
            return
        sd = _import_sounddevice()
        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=self._audio_callback,
            blocksize=1024,
            dtype=np.float32,
        )
        self.stream.start()

    def stop_stream(self):
        if self.stream:
            try:
                self.stream.stop()
            except Exception:
                pass
            try:
                self.stream.close()
            except Exception:
                pass
            self.stream = None

        self.is_playing = False
        with self._pause_lock:
            self._paused = False
        self.clear_queue()

    def cleanup(self):
        self.stop_stream()
        self.current_audio = None
        self.playback_complete_callback = None

    def play_audio(self, audio_array: np.ndarray):
        if audio_array is None or len(audio_array) == 0:
            return

        if audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32)

        max_abs = float(np.max(np.abs(audio_array))) if len(audio_array) else 0.0
        if max_abs > 1.0:
            audio_array = audio_array / max_abs

        self.audio_queue.put(audio_array)
        self.is_playing = True
        if self.stream is None:
            self.start_stream()

    def pause(self) -> bool:
        with self._pause_lock:
            if self.is_playing and not self._paused:
                self._paused = True
                if self.on_audio_pause:
                    threading.Thread(target=self.on_audio_pause, daemon=True).start()
                return True
        return False

    def resume(self) -> bool:
        with self._pause_lock:
            if self._paused:
                self._paused = False
                if self.on_audio_resume:
                    threading.Thread(target=self.on_audio_resume, daemon=True).start()
                return True
        return False

    def is_paused_state(self) -> bool:
        with self._pause_lock:
            return bool(self._paused)

    def clear_queue(self):
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        self.current_audio = None
        self.current_position = 0

