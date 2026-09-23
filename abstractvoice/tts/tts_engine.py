"""Core audio playback utilities.

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

logger = logging.getLogger(__name__)


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
            "  pip install \"abstractvoice[audio-io]\"\n"
            "  pip install \"abstractvoice[apple]\"  # Apple profile\n"
            "  pip install \"abstractvoice[gpu]\"    # GPU profile\n"
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

    def __init__(self, sample_rate: int = 22050, debug_mode: bool = False, output_device=None):
        self.sample_rate = int(sample_rate)
        self.debug_mode = debug_mode

        # WHICH SPEAKER. None/"" follows the system default; a UID (stable across
        # reboots), a device name, or a PortAudio index pins one device.
        self._output_device_spec = output_device
        # The device the open stream actually landed on, by stable key — what a
        # default change is compared against. An INDEX cannot do this job: indices
        # are positions in PortAudio's frozen list, so the same index can mean a
        # different device after a hotplug.
        self._opened_device_key: str | None = None
        # Called with a human sentence whenever playback cannot use the device it
        # was asked for. Silence here is what made this class of bug invisible:
        # audio came out of the wrong speaker and nothing anywhere said so.
        self.on_output_device_problem: Optional[Callable[[str], None]] = None
        self._reported_device_problems: set[str] = set()
        # Guard for the one operation that can disturb the rest of the process:
        # refreshing PortAudio's frozen device list invalidates every open stream,
        # including a microphone. Hosts that may be listening override this.
        self.allow_device_refresh: Optional[Callable[[], bool]] = None

        self.audio_queue: "queue.Queue[np.ndarray]" = queue.Queue()
        self.stream = None
        self._stream_lock = threading.RLock()
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
        """Move playback when the device it should use changed (idle-only).

        Compared by DEVICE IDENTITY, not by PortAudio's index. Two reasons the index
        could not do this job: it is a position in a list PortAudio froze at
        initialization, so it can name a different device after a hotplug; and the
        "default output" that list reports is itself the default from process start,
        so a user switching to headphones an hour in changed nothing this check could
        see. `resolve_output_device` asks the system what is true now.

        Still idle-only: restarting mid-sentence would cut the sentence.
        """
        if self.stream is None:
            return
        if not self._is_idle():
            return

        try:
            from .audio_devices import resolve_output_device

            wanted = resolve_output_device(self._output_device_spec)
        except Exception:
            return
        if wanted is None:
            return

        opened_key = getattr(self, "_opened_device_key", None)
        if opened_key is None:
            self._opened_device_key = wanted.key
            return
        if wanted.key == opened_key:
            return

        if self.debug_mode:
            try:
                old = str(getattr(self, "_opened_output_device_name", None) or "").strip() or "(unknown)"
                print(f"ℹ️  Audio output changed; restarting stream: {old} -> {wanted.name}")
            except Exception:
                pass

        # Record the new identity before attempting the move: a failed restart must
        # not re-attempt on every single utterance.
        self._opened_device_key = wanted.key
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

    def set_output_device(self, spec) -> None:
        """Choose the speaker. None/"" = follow the system default.

        Takes effect on the next utterance; an open idle stream is restarted at once
        so the choice is audible immediately rather than after the current reply.
        """
        self._output_device_spec = spec
        self._reported_device_problems.clear()
        with self._stream_lock:
            if self.stream is not None and self._is_idle():
                try:
                    self.stop_stream()
                except Exception:
                    return
                try:
                    self.start_stream()
                except Exception:
                    pass

    def get_output_device(self):
        return self._output_device_spec

    def current_output_device_name(self) -> str:
        """The device the open stream is on, or the one the next utterance will use."""
        name = str(getattr(self, "_opened_output_device_name", None) or "").strip()
        if name:
            return name
        try:
            from .audio_devices import resolve_output_device

            device = resolve_output_device(self._output_device_spec)
            return device.name if device is not None else ""
        except Exception:
            return ""

    def _report_device_problem(self, message: str) -> None:
        """Say it once per distinct problem, to the host and to the log."""
        text = str(message or "").strip()
        if not text or text in self._reported_device_problems:
            return
        self._reported_device_problems.add(text)
        logger.warning("#FALLBACK audio output: %s", text)
        if self.debug_mode:
            try:
                print(f"⚠️  {text}")
            except Exception:
                pass
        callback = self.on_output_device_problem
        if callable(callback):
            try:
                callback(text)
            except Exception:
                pass

    def _may_refresh_devices(self) -> bool:
        gate = self.allow_device_refresh
        if callable(gate):
            try:
                return bool(gate())
            except Exception:
                return False
        return self.stream is None

    def refresh_device_list(self) -> bool:
        """Re-enumerate audio devices (PortAudio caches them at initialization).

        A device connected after this process started is invisible until this runs —
        and running it invalidates EVERY open stream in the process, including a
        microphone, which is why it is gated and never automatic during playback.
        Measured cost: ~4 ms.
        """
        if not self._may_refresh_devices():
            return False
        sd = _import_sounddevice()
        try:
            with self._stream_lock:
                if self.stream is not None:
                    return False
                sd._terminate()
                sd._initialize()
            return True
        except Exception as e:
            if self.debug_mode:
                try:
                    print(f"⚠️  Could not refresh the audio device list: {e}")
                except Exception:
                    pass
            return False

    def _resolve_stream_targets(self) -> "list[tuple[int | None, str, str]]":
        """Ordered (portaudio_index, label, problem_to_report_first) to try.

        ONE target in the normal case: the device that was asked for. The old code
        built a candidate list of EVERY output device and walked it silently, so a
        default that failed to open sent speech to whatever came next in PortAudio's
        list — on a Mac, the built-in speakers. That is the bug this method exists to
        make impossible: a fallback still happens when the wanted device is genuinely
        gone, but it is deliberate, it is the SYSTEM DEFAULT rather than an arbitrary
        device, and it is reported.
        """
        try:
            from .audio_devices import resolve_output_device, system_default_output
        except Exception:
            return [(None, "system default", "")]

        spec = self._output_device_spec
        pinned = bool(spec is not None and str(spec).strip())
        targets: list[tuple[int | None, str, str]] = []

        device = resolve_output_device(spec)
        if device is not None and device.index is None and self._may_refresh_devices():
            # Known to the system but not to PortAudio: the list is stale (the device
            # was connected after this process started). This is the only automatic
            # refresh, and only ever with our own stream closed.
            if self.refresh_device_list():
                device = resolve_output_device(spec)

        if device is not None and device.index is not None:
            targets.append((int(device.index), device.name, ""))
        elif pinned:
            fallback = system_default_output()
            label = fallback.name if fallback is not None else "the system default"
            problem = (
                f"The selected audio output ({spec}) is not available, so speech is playing "
                f"on {label} instead."
            )
            if fallback is not None and fallback.index is not None:
                targets.append((int(fallback.index), fallback.name, problem))
            else:
                targets.append((None, label, problem))
        elif device is not None:
            # The system's output exists but PortAudio cannot see it, and a refresh was
            # refused (the microphone is open). `device=None` is all that is left — and
            # that means PortAudio's STARTUP default, i.e. exactly the device the user is
            # not listening to. Playing there silently, under a label naming the device we
            # are NOT using, is the original bug with better manners: say it instead.
            targets.append(
                (
                    None,
                    "the audio engine's startup device",
                    f"Speech cannot be sent to {device.name} yet: it appeared after this app "
                    f"started, and the audio engine cannot be refreshed while the microphone "
                    f"is open. Playing on the startup device instead.",
                )
            )
        else:
            # Nothing resolvable at all (no CoreAudio, and PortAudio could not be queried).
            # There is no evidence anything is wrong, so let PortAudio choose, quietly.
            targets.append((None, "the system default", ""))
        return targets

    def start_stream(self):
        with self._stream_lock:
            if self.stream is not None:
                try:
                    if self.stream.active:
                        return
                except Exception:
                    pass  # A disconnected/closed device can reject the query.
                # CoreAudio can stop a stream without destroying its Python
                # object (sleep, USB interruption). Reusing it queues audio
                # forever: no output callback will consume those samples.
                logger.warning("#FALLBACK audio output: reopening an inactive stream")
                try:
                    self.stream.close()
                except Exception as exc:
                    logger.warning("#FALLBACK audio output: stale stream close failed: %s", exc)
                finally:
                    self.stream = None
                self._audio_started = False
                # Preserve pending audio and pause state across recovery.
            sd = _import_sounddevice()

            desired_sr = int(self.sample_rate)

            targets = self._resolve_stream_targets()

            # Common output rates (keep short; we already prefer device default + desired).
            common_rates = (48000, 44100, 24000, 22050, 16000)

            last_err: Exception | None = None
            for device, target_label, target_problem in targets:
                if target_problem:
                    self._report_device_problem(target_problem)
                # Build per-device candidate sample rates. Prefer the hardware
                # default first. Some CoreAudio devices take several seconds to
                # reject uncommon rates (24 kHz/22.05 kHz); resampling is cheaper
                # and avoids a slow first `/speak`.
                sr_candidates: list[int] = []
                try:
                    dev = sd.query_devices(device, "output") if device is not None else sd.query_devices(None, "output")
                    default_sr = int(round(float(dev.get("default_samplerate", 0) or 0)))
                    max_ch = int(dev.get("max_output_channels", 0) or 0)
                except Exception:
                    default_sr = 0
                    max_ch = 0

                if default_sr:
                    sr_candidates.append(int(default_sr))
                if desired_sr and int(desired_sr) not in sr_candidates:
                    sr_candidates.append(int(desired_sr))

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
                                try:
                                    # The stable identity of what we are actually playing on,
                                    # so a later default change is compared by DEVICE and not
                                    # by a PortAudio index that can mean something else now.
                                    from .audio_devices import resolve_output_device

                                    opened = resolve_output_device(
                                        int(device) if device is not None else None
                                    )
                                    self._opened_device_key = opened.key if opened is not None else None
                                except Exception:
                                    self._opened_device_key = None
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
                                    print(f"ℹ️  Audio output opened at {sr}Hz; playback will resample from {desired_sr}Hz")
                                return
                            except Exception as e:
                                last_err = e
                                try:
                                    if stream is not None:
                                        stream.close()
                                except Exception:
                                    pass
                                continue

            # Every target failed. Name the device that could not be opened: the
            # caller's only other signal is silence.
            wanted = targets[0][1] if targets else "the system default"
            self._report_device_problem(
                f"Could not open the audio output ({wanted}): {last_err}"
                if last_err is not None
                else f"Could not open the audio output ({wanted})."
            )
            if last_err is not None:
                raise last_err
            raise RuntimeError(f"Failed to start audio output stream on {wanted}")

    def stop_stream(self):
        with self._stream_lock:
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
        self._opened_device_key = None

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
            self.start_stream()  # Also recovers a surviving but inactive stream.

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
