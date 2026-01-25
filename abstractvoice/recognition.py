"""Voice recognition module that combines VAD and STT."""

import threading
import time
from typing import Optional
from collections import deque

import numpy as np
import re

from .stop_phrase import is_stop_phrase
from .audio.resample import linear_resample_mono

# Lazy imports for heavy dependencies
def _import_audio_deps():
    """Import audio dependencies with helpful error message if missing."""
    try:
        import sounddevice as sd
        return sd
    except ImportError as e:
        raise ImportError(
            "Audio capture/playback requires sounddevice. Install with:\n"
            "  pip install abstractvoice          # Core install (includes sounddevice)\n"
            "  pip install abstractvoice[all]      # All features\n"
            f"Original error: {e}"
        ) from e

def _import_vad():
    """Import VoiceDetector with helpful error message if dependencies missing."""
    try:
        from .vad import VoiceDetector
        return VoiceDetector
    except ImportError as e:
        if "webrtcvad" in str(e):
            raise ImportError(
                "Voice activity detection requires optional dependencies. Install with:\n"
                "  pip install abstractvoice[voice]  # For basic audio\n"
                "  pip install abstractvoice[all]    # For all features\n"
                f"Original error: {e}"
            ) from e
        raise

def _import_transcriber():
    """Import STT adapter with helpful error message if dependencies missing."""
    try:
        from .adapters.stt_faster_whisper import FasterWhisperAdapter
        return FasterWhisperAdapter
    except ImportError as e:
        raise ImportError(
            "Speech recognition requires faster-whisper (core dependency). "
            "If this error occurs, your installation is inconsistent.\n"
            "Try reinstalling:\n"
            "  pip install --upgrade abstractvoice\n"
            f"Original error: {e}"
        ) from e
        raise


def _import_aec_processor():
    """Import AEC processor with helpful error if dependencies missing."""
    try:
        from .aec.webrtc_apm import AecConfig, WebRtcAecProcessor
        return AecConfig, WebRtcAecProcessor
    except ImportError as e:
        raise ImportError(
            "AEC is optional and requires extra dependencies.\n"
            "Install with: pip install \"abstractvoice[aec]\"\n"
            f"Original error: {e}"
        ) from e


class VoiceRecognizer:
    """Voice recognition with VAD and STT."""
    
    def __init__(self, transcription_callback, stop_callback=None, 
                 vad_aggressiveness=1, min_speech_duration=600, 
                 silence_timeout=1500, sample_rate=16000, 
                 chunk_duration=30, whisper_model="tiny", 
                 min_transcription_length=5, debug_mode=False,
                 aec_enabled: bool = False, aec_stream_delay_ms: int = 0,
                 language: str | None = None,
                 allow_downloads: bool = True):
        """Initialize voice recognizer.
        
        Args:
            transcription_callback: Function to call with transcription text
            stop_callback: Function to call when "stop" is detected
            vad_aggressiveness: VAD aggressiveness (0-3)
            min_speech_duration: Min speech duration in ms to start recording
            silence_timeout: Silence timeout in ms to end recording
            sample_rate: Audio sample rate in Hz
            chunk_duration: Audio chunk duration in ms
            whisper_model: Whisper model name
            min_transcription_length: Min valid transcription length
            debug_mode: Enable debug output
        """
        self.debug_mode = debug_mode
        self.transcription_callback = transcription_callback
        self.stop_callback = stop_callback
        self.language = (language or None)
        self.allow_downloads = bool(allow_downloads)

        # Stop phrase(s): robust “interrupt” without requiring echo cancellation.
        # Keep it conservative to avoid accidental stops from the assistant audio.
        # Include bare "stop" because users will naturally say it.
        self.stop_phrases = ["stop", "ok stop", "okay stop"]

        # While TTS is playing we can end up with continuous "speech" from speaker echo,
        # which prevents end-of-utterance detection and therefore prevents stop phrase
        # transcription. To keep STOP mode usable without AEC, we run a low-rate rolling
        # window transcription ONLY for stop-phrase detection when transcriptions are paused.
        self._stop_ring = bytearray()
        self._stop_last_check = 0.0
        # Faster checks help catch "ok stop" early during playback.
        self._stop_check_interval_s = 0.6
        self._stop_window_s = 2.0
        self._stop_hit_count = 0
        self._stop_hit_deadline = 0.0
        
        # Configuration
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration  # in ms
        self.chunk_size = int(sample_rate * chunk_duration / 1000)
        self.min_speech_chunks = int(min_speech_duration / chunk_duration)
        self.silence_timeout_chunks = int(silence_timeout / chunk_duration)
        self._default_min_speech_chunks = int(self.min_speech_chunks)
        self._default_silence_timeout_chunks = int(self.silence_timeout_chunks)
        
        # Initialize components using lazy imports
        VoiceDetector = _import_vad()
        self.voice_detector = VoiceDetector(
            aggressiveness=vad_aggressiveness,
            sample_rate=sample_rate,
            debug_mode=debug_mode
        )

        # STT: use faster-whisper adapter by default (core dependency)
        STTAdapter = _import_transcriber()
        self.stt_adapter = STTAdapter(
            model_size=whisper_model,
            device="auto",
            compute_type="int8",
            allow_downloads=bool(self.allow_downloads),
        )
        self.min_transcription_length = min_transcription_length
        
        # State
        self.is_running = False
        self.thread = None
        self.stream = None
        self.tts_interrupt_callback = None
        self.tts_interrupt_enabled = True  # Can be disabled during TTS playback
        self.listening_paused = False  # Can be paused to completely stop processing audio
        # While TTS is playing (esp. without AEC), we often want to suppress normal
        # transcriptions to avoid self-feedback loops, but still allow stop phrase.
        self.transcriptions_paused = False
        self._profile = "stop"

        # Optional AEC (echo cancellation) state.
        self.aec_enabled = False
        self._aec = None
        self._far_end_lock = threading.Lock()
        self._far_end_pcm16 = bytearray()
        if aec_enabled:
            self.enable_aec(True, stream_delay_ms=aec_stream_delay_ms)

        # Apply initial profile.
        self.set_profile("stop")

    def set_profile(self, profile: str) -> None:
        """Set listening profile tuned for the current interaction mode.

        Why this exists:
        - PTT needs *very* low thresholds to reliably capture short utterances.
        - STOP/WAIT should use more conservative defaults to reduce false triggers.
        """
        p = (profile or "").strip().lower()
        if p not in ("stop", "wait", "full", "ptt"):
            return
        self._profile = p

        if p == "ptt":
            # Make capture responsive: start recording as soon as we see speech,
            # and end quickly after short silence.
            self.min_speech_chunks = 1
            # ~700ms of silence to end (tuned for quick PTT turns).
            self.silence_timeout_chunks = max(8, int(round(700.0 / float(self.chunk_duration))))
            self.transcriptions_paused = False
            self.listening_paused = False
            return

        if p == "full":
            # Make FULL responsive: start recording sooner, end sooner.
            # This improves "didn't recognize me" reports on headsets.
            self.min_speech_chunks = max(3, int(round(180.0 / float(self.chunk_duration))))
            self.silence_timeout_chunks = max(12, int(round(900.0 / float(self.chunk_duration))))
            return

        # Default/conservative for continuous modes.
        self.min_speech_chunks = int(self._default_min_speech_chunks)
        self.silence_timeout_chunks = int(self._default_silence_timeout_chunks)

    def enable_aec(self, enabled: bool = True, *, stream_delay_ms: int = 0) -> bool:
        """Enable/disable acoustic echo cancellation (optional).

        When enabled, the recognizer expects far-end audio via `feed_far_end_audio()`.
        """
        if not enabled:
            self.aec_enabled = False
            self._aec = None
            with self._far_end_lock:
                self._far_end_pcm16 = bytearray()
            return True

        AecConfig, WebRtcAecProcessor = _import_aec_processor()
        self._aec = WebRtcAecProcessor(
            AecConfig(sample_rate=int(self.sample_rate), channels=1, stream_delay_ms=int(stream_delay_ms))
        )
        self.aec_enabled = True
        return True

    def feed_far_end_audio(self, audio_chunk: np.ndarray, *, sample_rate: int) -> None:
        """Provide far-end (speaker) audio reference for AEC.

        audio_chunk: mono float32 in [-1, 1] (as written to speaker output)
        """
        if not self.aec_enabled:
            return
        if audio_chunk is None or len(audio_chunk) == 0:
            return

        mono = audio_chunk.astype(np.float32, copy=False)
        if int(sample_rate) != int(self.sample_rate):
            mono = linear_resample_mono(mono, int(sample_rate), int(self.sample_rate))

        pcm16 = np.clip(mono, -1.0, 1.0)
        pcm16 = (pcm16 * 32767.0).astype(np.int16).tobytes()

        with self._far_end_lock:
            self._far_end_pcm16.extend(pcm16)

    def _pop_far_end_pcm16(self, nbytes: int) -> bytes:
        if nbytes <= 0:
            return b""
        with self._far_end_lock:
            if not self._far_end_pcm16:
                return b"\x00" * nbytes
            take = min(nbytes, len(self._far_end_pcm16))
            out = bytes(self._far_end_pcm16[:take])
            del self._far_end_pcm16[:take]
        if take < nbytes:
            out += b"\x00" * (nbytes - take)
        return out

    def _apply_aec(self, near_pcm16: bytes) -> bytes:
        if not (self.aec_enabled and self._aec):
            return near_pcm16

        # The underlying APM typically expects 10ms frames. We can split any chunk
        # size into 10ms sub-frames for robustness.
        frame_bytes = int(self.sample_rate * 0.01) * 2  # 10ms * int16
        if frame_bytes <= 0:
            return near_pcm16
        if len(near_pcm16) % frame_bytes != 0:
            # Pad to whole frames.
            pad = frame_bytes - (len(near_pcm16) % frame_bytes)
            near_pcm16 = near_pcm16 + (b"\x00" * pad)

        out = bytearray()
        for i in range(0, len(near_pcm16), frame_bytes):
            near = near_pcm16[i : i + frame_bytes]
            far = self._pop_far_end_pcm16(frame_bytes)
            out.extend(self._aec.process(near_pcm16=near, far_pcm16=far))
        return bytes(out)
    
    def start(self, tts_interrupt_callback=None):
        """Start voice recognition in a separate thread.
        
        Args:
            tts_interrupt_callback: Function to call when speech is detected during listening
            
        Returns:
            True if started, False if already running
        """
        if self.is_running:
            return False
        
        self.tts_interrupt_callback = tts_interrupt_callback
        self.is_running = True
        self.thread = threading.Thread(target=self._recognition_loop)
        self.thread.start()
        
        if self.debug_mode:
            print(" > Voice recognition started")
        return True
    
    def stop(self):
        """Stop voice recognition.
        
        Returns:
            True if stopped, False if not running
        """
        if not self.is_running:
            return False
        
        self.is_running = False
        if self.thread:
            self.thread.join()
        
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
        
        if self.debug_mode:
            print(" > Voice recognition stopped")
        return True

    def _transcribe_pcm16(
        self,
        pcm16_bytes: bytes,
        language: Optional[str] = None,
        *,
        hotwords: str | None = None,
        condition_on_previous_text: bool = True,
    ) -> str:
        """Transcribe raw PCM16 mono audio bytes."""
        if not pcm16_bytes:
            return ""

        audio = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        lang = language if language is not None else self.language
        text = self.stt_adapter.transcribe_from_array(
            audio,
            sample_rate=self.sample_rate,
            language=lang,
            hotwords=hotwords,
            condition_on_previous_text=bool(condition_on_previous_text),
        )
        return (text or "").strip()

    def _is_stop_command(self, text: str) -> bool:
        """Return True if text matches a configured stop phrase."""
        return is_stop_phrase(text, self.stop_phrases)

    def _match_stop_phrase(self, text: str) -> str | None:
        """Return the matched stop phrase (normalized) or None."""
        from .stop_phrase import normalize_stop_phrase

        normalized = normalize_stop_phrase(text)
        if not normalized:
            return None
        phrases = [normalize_stop_phrase(p) for p in (self.stop_phrases or []) if p]
        for ph in phrases:
            if not ph:
                continue
            if normalized == ph or normalized.startswith(ph + " ") or normalized.endswith(" " + ph):
                return ph
        return None

    def _maybe_detect_stop_phrase_continuous(self, pcm16_chunk: bytes) -> bool:
        """Best-effort rolling stop-phrase detection during TTS playback.

        Returns True if stop_callback was invoked.
        """
        if not (self.transcriptions_paused and self.stop_callback):
            return False

        now = time.time()
        self._stop_ring.extend(pcm16_chunk)
        max_bytes = int(self.sample_rate * float(self._stop_window_s) * 2)
        if max_bytes > 0 and len(self._stop_ring) > max_bytes:
            del self._stop_ring[: len(self._stop_ring) - max_bytes]

        if (now - float(self._stop_last_check)) < float(self._stop_check_interval_s):
            return False
        self._stop_last_check = now

        try:
            text = self._transcribe_pcm16(
                bytes(self._stop_ring),
                hotwords="stop, ok stop, okay stop",
                condition_on_previous_text=False,
            )
        except Exception:
            return False

        # Keep this conservative to avoid hallucinated "stop" from hotword bias:
        # - only accept short transcripts
        # - require confirmation for bare "stop"
        words = (text or "").strip().split()
        if len(words) > 4:
            self._stop_hit_count = 0
            return False

        matched = self._match_stop_phrase(text or "")
        if matched:
            now2 = time.time()
            # Confirmation: for bare "stop" require 2 hits within 2.5s.
            if matched == "stop":
                if now2 > float(self._stop_hit_deadline):
                    self._stop_hit_count = 0
                self._stop_hit_deadline = now2 + 2.5
                self._stop_hit_count += 1
                if self._stop_hit_count < 2:
                    return False
            else:
                self._stop_hit_count = 0

            try:
                self.stop_callback()
            except Exception:
                pass
            self._stop_ring = bytearray()
            # small cooldown
            self._stop_last_check = time.time()
            return True
        return False
    
    def _recognition_loop(self):
        """Main recognition loop."""
        sd = _import_audio_deps()

        # NOTE: sounddevice uses PortAudio under the hood (same as our TTS playback).
        # Keeping microphone capture in-process avoids PyAudio install issues.
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=self.chunk_size,
        )
        self.stream.start()
        
        speech_buffer = []
        speech_count = 0
        silence_count = 0
        recording = False
        
        while self.is_running:
            try:
                # If listening is paused, sleep briefly and skip processing
                if self.listening_paused:
                    time.sleep(0.1)
                    continue
                
                # Read audio data
                audio_chunk, overflowed = self.stream.read(self.chunk_size)
                if overflowed and self.debug_mode:
                    print(" > Mic input overflow")
                audio_data = audio_chunk.tobytes()

                # Optional AEC: remove speaker echo from mic input before VAD/STT.
                if self.aec_enabled and self._aec:
                    audio_data = self._apply_aec(audio_data)

                # While transcriptions are paused (typically during TTS in STOP mode),
                # run a rolling stop-phrase detector so "stop" can still work even if
                # VAD never sees a clean end-of-utterance due to speaker echo.
                if self._maybe_detect_stop_phrase_continuous(audio_data):
                    # Don't also feed this chunk into VAD/recording state.
                    continue
                
                # Check for speech
                is_speech = self.voice_detector.is_speech(audio_data)
                
                if is_speech:
                    speech_buffer.append(audio_data)
                    speech_count += 1
                    silence_count = 0
                    
                    # Trigger TTS interrupt callback if enough speech detected
                    # Only interrupt if TTS interruption is enabled (not during TTS playback)
                    if (self.tts_interrupt_callback and 
                        self.tts_interrupt_enabled and
                        speech_count >= self.min_speech_chunks and 
                        not recording):
                        self.tts_interrupt_callback()
                        if self.debug_mode:
                            print(" > TTS interrupted by user speech")
                    
                    # Start recording after minimum speech detected
                    if speech_count >= self.min_speech_chunks:
                        recording = True
                        
                else:
                    # Handle silence during recording
                    if recording:
                        speech_buffer.append(audio_data)
                        silence_count += 1
                        
                        # End of speech detected
                        if silence_count >= self.silence_timeout_chunks:
                            if self.debug_mode:
                                print(f" > Speech detected ({len(speech_buffer)} chunks), transcribing...")
                                
                            audio_bytes = b''.join(speech_buffer)
                            text = self._transcribe_pcm16(audio_bytes)
                            
                            if text:
                                # Check for stop command
                                if self._is_stop_command(text):
                                    if self.stop_callback:
                                        self.stop_callback()
                                    else:
                                        # If no stop callback, invoke transcription callback anyway
                                        self.transcription_callback(text)
                                else:
                                    # Normal transcription (can be suppressed during TTS)
                                    if not self.transcriptions_paused:
                                        self.transcription_callback(text)
                            
                            # Reset state
                            speech_buffer = []
                            speech_count = 0
                            silence_count = 0
                            recording = False
                    else:
                        # No speech detected and not recording
                        speech_count = max(0, speech_count - 1)
                        if speech_count == 0:
                            speech_buffer = []
                            
            except Exception as e:
                if self.debug_mode:
                    print(f"Voice recognition error: {e}")
                continue
    
    def change_whisper_model(self, model_name):
        """Change the Whisper model.
        
        Args:
            model_name: New model name
            
        Returns:
            True if changed, False otherwise
        """
        try:
            # Recreate adapter to switch model size.
            STTAdapter = _import_transcriber()
            self.stt_adapter = STTAdapter(model_size=model_name, device="cpu", compute_type="int8")
            return True
        except Exception as e:
            if self.debug_mode:
                print(f"STT model change error: {e}")
            return False
    
    def change_vad_aggressiveness(self, aggressiveness):
        """Change VAD aggressiveness.
        
        Args:
            aggressiveness: New aggressiveness level (0-3)
            
        Returns:
            True if changed, False otherwise
        """
        return self.voice_detector.set_aggressiveness(aggressiveness)
    
    def pause_tts_interrupt(self):
        """Temporarily disable TTS interruption (e.g., during TTS playback).
        
        This prevents the system from interrupting its own speech.
        """
        self.tts_interrupt_enabled = False
        if self.debug_mode:
            print(" > TTS interrupt paused")
    
    def resume_tts_interrupt(self):
        """Re-enable TTS interruption after it was paused."""
        self.tts_interrupt_enabled = True
        if self.debug_mode:
            print(" > TTS interrupt resumed")
    
    def pause_listening(self):
        """Temporarily pause audio processing entirely (e.g., during TTS in 'wait' mode).
        
        This completely stops processing audio input while keeping the thread alive.
        """
        self.listening_paused = True
        if self.debug_mode:
            print(" > Listening paused")
    
    def resume_listening(self):
        """Resume audio processing after it was paused."""
        self.listening_paused = False
        if self.debug_mode:
            print(" > Listening resumed") 

    def pause_transcriptions(self):
        """Suppress normal transcriptions while still allowing stop phrase detection."""
        self.transcriptions_paused = True
        if self.debug_mode:
            print(" > Transcriptions paused")

    def resume_transcriptions(self):
        """Re-enable normal transcriptions after they were suppressed."""
        self.transcriptions_paused = False
        if self.debug_mode:
            print(" > Transcriptions resumed")