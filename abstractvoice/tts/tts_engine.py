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

    Prefers librosa when installed; otherwise falls back to a lightweight WSOLA
    implementation so `/speed` works without extra deps.
    """
    if not speed or speed == 1.0:
        return audio

    try:
        librosa = _import_librosa()
        return librosa.effects.time_stretch(np.asarray(audio, dtype=np.float32).reshape(-1), rate=float(speed))
    except ImportError:
        try:
            from .time_stretch import wsola_time_stretch

            return wsola_time_stretch(np.asarray(audio, dtype=np.float32).reshape(-1), rate=float(speed), sr=int(sr))
        except Exception as e:
            logging.warning(f"Time-stretching failed: {e}, using original audio")
            return audio
    except Exception as e:
        try:
            from .time_stretch import wsola_time_stretch

            return wsola_time_stretch(np.asarray(audio, dtype=np.float32).reshape(-1), rate=float(speed), sr=int(sr))
        except Exception:
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

        # Track which output device the stream is pinned to.
        #
        # Why: on macOS it's common for the system "default output" device to change
        # while the process is running (AirPods connect/disconnect, monitor audio,
        # etc.). We keep the PortAudio stream open for stability, but that can pin
        # playback to a now-non-default device. Users then observe that system volume
        # keys appear to “not work” (they control the *current* default device).
        #
        # We track the device we opened and will (best-effort) restart the stream
        # when the system default output changes and the stream is idle.
        self._opened_output_device_index: int | None = None
        self._opened_output_device_name: str | None = None
        self._opened_output_channels: int | None = None
        self._last_seen_default_output_device_index: int | None = None

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

    def _default_output_device_info(self):
        """Best-effort query of the current system default output device."""
        sd = _import_sounddevice()
        try:
            return sd.query_devices(None, "output")
        except Exception:
            return None

    def _default_output_device_index(self) -> int | None:
        info = self._default_output_device_info()
        if not isinstance(info, dict):
            return None
        idx = info.get("index", None)
        return int(idx) if isinstance(idx, int) else None

    def _is_idle(self) -> bool:
        """Return True when the stream is open but no audio is draining."""
        if bool(getattr(self, "is_playing", False)):
            return False
        try:
            if not self.audio_queue.empty():
                return False
        except Exception:
            # If we can't inspect the queue, be conservative.
            return False
        try:
            ca = getattr(self, "current_audio", None)
            pos = int(getattr(self, "current_position", 0) or 0)
            if ca is None:
                return True
            try:
                n = int(len(ca))
            except Exception:
                n = 0
            return bool(pos >= n)
        except Exception:
            return False

    def _maybe_restart_stream_for_default_device_change(self) -> None:
        """If the OS default output device changed, restart the stream (idle-only).

        This is best-effort and intentionally conservative: we only restart when
        the stream is idle to avoid audible glitches or races.
        """
        if self.stream is None:
            return
        if not self._is_idle():
            return

        current_default = self._default_output_device_index()
        if current_default is None:
            return

        last_seen = getattr(self, "_last_seen_default_output_device_index", None)
        if last_seen is None:
            self._last_seen_default_output_device_index = int(current_default)
            return

        if int(current_default) == int(last_seen):
            return

        # Default output changed since last time we observed it.
        if self.debug_mode:
            try:
                old = str(getattr(self, "_opened_output_device_name", None) or "").strip() or "(unknown)"
                info = self._default_output_device_info() or {}
                new = str(info.get("name", "") or "").strip() or f"index={int(current_default)}"
                print(f"ℹ️  Default output changed; restarting audio stream: {old} -> {new}")
            except Exception:
                pass

        # Try to restart on the new default. Even if this fails and we fall back
        # to a different device, we update `last_seen` so we don't repeatedly
        # attempt restarts until the default changes again.
        self._last_seen_default_output_device_index = int(current_default)
        try:
            self.stop_stream()
        except Exception:
            # Best-effort; if stop fails, keep the existing stream.
            return
        try:
            self.start_stream()
        except Exception as e:
            if self.debug_mode:
                try:
                    print(f"⚠️  Failed to restart audio stream after device change: {e}")
                except Exception:
                    pass

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

        # Device candidates: default first, then explicit indices as fallback.
        device_candidates: list[int | None] = [None]
        try:
            dev = sd.query_devices(None, "output")  # default output device
            idx = dev.get("index", None)
            if isinstance(idx, int) and idx not in device_candidates:
                device_candidates.append(idx)
        except Exception:
            pass
        try:
            for i, dev in enumerate(sd.query_devices()):  # all devices
                if int(dev.get("max_output_channels", 0) or 0) <= 0:
                    continue
                if i not in device_candidates:
                    device_candidates.append(i)
        except Exception:
            pass

        # Common output rates (keep short; we already prefer device default + desired).
        common_rates = (48000, 44100, 24000, 22050, 16000)

        last_err: Exception | None = None
        for device in device_candidates:
            # Build per-device candidate sample rates (desired first, then device default, then common).
            sr_candidates: list[int] = []
            try:
                dev = sd.query_devices(device, "output") if device is not None else sd.query_devices(None, "output")
                default_sr = int(round(float(dev.get("default_samplerate", 0) or 0)))
                max_ch = int(dev.get("max_output_channels", 0) or 0)
            except Exception:
                default_sr = 0
                max_ch = 0

            # Prefer opening the stream at the audio's natural rate to avoid resampling
            # (e.g. 24 kHz for AudioDiT). If the device rejects it, we fall back.
            if desired_sr:
                sr_candidates.append(int(desired_sr))
            if default_sr and int(default_sr) not in sr_candidates:
                sr_candidates.append(int(default_sr))

            for sr in common_rates:
                if sr not in sr_candidates:
                    sr_candidates.append(sr)

            # Prefer stereo devices (most macOS outputs are 2ch); fall back to mono.
            ch_order = (2, 1) if max_ch >= 2 else (1,)
            # Prefer PortAudio-chosen blocksize first (often most compatible).
            block_order = (0, 1024)

            for sr in sr_candidates:
                for blocksize in block_order:
                    for channels in ch_order:
                        if max_ch and channels > max_ch:
                            continue
                        stream = None
                        try:
                            with _SilenceStderrFD(enabled=not self.debug_mode):
                                stream = sd.OutputStream(
                                    samplerate=int(sr),
                                    channels=int(channels),
                                    callback=self._audio_callback,
                                    blocksize=int(blocksize),
                                    dtype=np.float32,
                                    device=int(device) if device is not None else None,
                                )
                                stream.start()
                            self.stream = stream
                            self.sample_rate = int(sr)
                            # Record which device we pinned to at open time.
                            try:
                                opened_info = (
                                    sd.query_devices(int(device), "output")
                                    if device is not None
                                    else sd.query_devices(None, "output")
                                )
                            except Exception:
                                opened_info = None
                            try:
                                if isinstance(opened_info, dict):
                                    self._opened_output_device_index = (
                                        int(opened_info.get("index")) if isinstance(opened_info.get("index"), int) else None
                                    )
                                    self._opened_output_device_name = str(opened_info.get("name", "") or "").strip() or None
                                else:
                                    self._opened_output_device_index = int(device) if device is not None else None
                                    self._opened_output_device_name = None
                            except Exception:
                                self._opened_output_device_index = int(device) if device is not None else None
                                self._opened_output_device_name = None
                            try:
                                self._opened_output_channels = int(channels)
                            except Exception:
                                self._opened_output_channels = None
                            try:
                                # Cache the system default output at open time so we can detect changes later.
                                cur_default = self._default_output_device_index()
                                self._last_seen_default_output_device_index = (
                                    int(cur_default) if cur_default is not None else self._last_seen_default_output_device_index
                                )
                            except Exception:
                                pass
                            if self.debug_mode:
                                try:
                                    name = str(getattr(self, "_opened_output_device_name", None) or "").strip() or "(unknown)"
                                    idx_txt = (
                                        str(int(self._opened_output_device_index))
                                        if isinstance(getattr(self, "_opened_output_device_index", None), int)
                                        else "?"
                                    )
                                    ch_txt = (
                                        str(int(self._opened_output_channels))
                                        if isinstance(getattr(self, "_opened_output_channels", None), int)
                                        else "?"
                                    )
                                    print(f"Audio output device: {name} (index={idx_txt}, channels={ch_txt}, sr={int(sr)}Hz)")
                                except Exception:
                                    pass
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

        # Clear device tracking (we will repopulate on next start).
        self._opened_output_device_index = None
        self._opened_output_device_name = None
        self._opened_output_channels = None

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
        # Defensive: never send NaN/Inf to the audio device.
        try:
            if not np.isfinite(audio_array).all():
                audio_array = np.nan_to_num(audio_array, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        except Exception:
            pass

        # If we haven't started the output stream yet, do so first. This allows
        # `start_stream()` to fall back to a compatible device sample rate.
        if self.stream is None:
            self.start_stream()
        else:
            # If the system default output changed since we opened the stream,
            # restart the stream when idle so system volume keys match what users
            # expect (best-effort).
            self._maybe_restart_stream_for_default_device_change()
            if self.stream is None:
                self.start_stream()

        sr_in = int(sample_rate) if sample_rate is not None else int(self.sample_rate)
        sr_out = int(self.sample_rate)
        if sr_in != sr_out:
            audio_array = linear_resample_mono(audio_array, sr_in, sr_out)

        max_abs = float(np.max(np.abs(audio_array))) if len(audio_array) else 0.0
        if max_abs > 1.0:
            audio_array = audio_array / max_abs
        else:
            # Best-effort clamp to avoid rare device-specific clipping issues.
            try:
                audio_array = np.clip(audio_array, -1.0, 1.0, out=audio_array)
            except Exception:
                pass

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
