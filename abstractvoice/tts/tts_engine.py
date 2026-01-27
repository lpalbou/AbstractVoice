"""Core audio playback utilities (Piper-first).

AbstractVoice core intentionally avoids shipping legacy Coqui-based TTSEngine
logic. This module contains only reusable audio utilities:
- `NonBlockingAudioPlayer` for low-latency pause/resume/stop
- `apply_speed_without_pitch_change` (optional librosa)
"""

from __future__ import annotations

import logging
import os
import queue
import threading
from typing import Callable, Optional

import numpy as np

from ..audio.resample import linear_resample_mono


_STDERR_FD_LOCK = threading.Lock()


class _SilenceStderrFD:
    """Temporarily redirect OS-level stderr (fd=2) to /dev/null.

    PortAudio (and some underlying CoreAudio/AUHAL code paths) can emit warnings
    directly to stderr, bypassing Python's `sys.stderr`. In interactive REPL
    contexts this can corrupt the prompt/spinner UI.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = bool(enabled)
        self._devnull_fd = None
        self._saved_stderr_fd = None

    def __enter__(self):
        if not self.enabled:
            return self
        _STDERR_FD_LOCK.acquire()
        try:
            self._devnull_fd = os.open(os.devnull, os.O_WRONLY)
            self._saved_stderr_fd = os.dup(2)
            os.dup2(self._devnull_fd, 2)
        except Exception:
            self.__exit__(None, None, None)
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self.enabled:
            return False
        try:
            if self._saved_stderr_fd is not None:
                try:
                    os.dup2(self._saved_stderr_fd, 2)
                except Exception:
                    pass
        finally:
            try:
                if self._saved_stderr_fd is not None:
                    os.close(self._saved_stderr_fd)
            except Exception:
                pass
            try:
                if self._devnull_fd is not None:
                    os.close(self._devnull_fd)
            except Exception:
                pass
            try:
                _STDERR_FD_LOCK.release()
            except Exception:
                pass
        return False


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

        desired_sr = int(self.sample_rate)
        candidates: list[int] = [desired_sr]
        try:
            dev = sd.query_devices(None, "output")  # default output device
            default_sr = int(round(float(dev.get("default_samplerate", 0) or 0)))
            if default_sr and default_sr not in candidates:
                candidates.append(default_sr)
        except Exception:
            default_sr = 0

        # Common output rates (keep short; we already prefer desired/default).
        for sr in (48000, 44100, 24000, 22050, 16000):
            if sr not in candidates:
                candidates.append(sr)

        last_err: Exception | None = None
        for sr in candidates:
            for blocksize in (1024, 0):  # 0 => PortAudio decides (often most compatible)
                stream = None
                try:
                    with _SilenceStderrFD(enabled=not self.debug_mode):
                        stream = sd.OutputStream(
                            samplerate=int(sr),
                            channels=1,
                            callback=self._audio_callback,
                            blocksize=int(blocksize),
                            dtype=np.float32,
                        )
                        stream.start()
                    self.stream = stream
                    self.sample_rate = int(sr)
                    if self.debug_mode and int(sr) != desired_sr:
                        print(f"⚠️  Output device rejected {desired_sr}Hz; using {sr}Hz (resampling)")
                    return
                except Exception as e:
                    last_err = e
                    try:
                        if stream is not None:
                            stream.close()
                    except Exception:
                        pass
                    continue

        # If we couldn't start, surface the last error.
        if last_err is not None:
            raise last_err
        raise RuntimeError("Failed to start audio output stream")

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

    def play_audio(self, audio_array: np.ndarray, *, sample_rate: int | None = None):
        if audio_array is None or len(audio_array) == 0:
            return

        # Ensure mono float32 vector.
        try:
            if hasattr(audio_array, "ndim") and int(audio_array.ndim) > 1:
                audio_array = np.mean(audio_array, axis=1).astype(np.float32)
        except Exception:
            pass
        audio_array = np.asarray(audio_array, dtype=np.float32).reshape(-1)

        # If we haven't started the output stream yet, do so first. This allows
        # `start_stream()` to fall back to a compatible device sample rate.
        if self.stream is None:
            self.start_stream()

        sr_in = int(sample_rate) if sample_rate is not None else int(self.sample_rate)
        sr_out = int(self.sample_rate)
        if sr_in != sr_out:
            audio_array = linear_resample_mono(audio_array, sr_in, sr_out)

        max_abs = float(np.max(np.abs(audio_array))) if len(audio_array) else 0.0
        if max_abs > 1.0:
            audio_array = audio_array / max_abs

        self.audio_queue.put(audio_array)
        self.is_playing = True
        # Stream should already be started above when needed.

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
