"""Voice recognition module that combines VAD and STT."""

import threading
import time
from typing import Optional

import numpy as np
import re

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


class VoiceRecognizer:
    """Voice recognition with VAD and STT."""
    
    def __init__(self, transcription_callback, stop_callback=None, 
                 vad_aggressiveness=1, min_speech_duration=600, 
                 silence_timeout=1500, sample_rate=16000, 
                 chunk_duration=30, whisper_model="tiny", 
                 min_transcription_length=5, debug_mode=False):
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

        # Stop phrase(s): robust “interrupt” without requiring echo cancellation.
        # Keep it conservative to avoid accidental stops from the assistant audio.
        self.stop_phrases = ["ok stop", "okay stop"]
        
        # Configuration
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration  # in ms
        self.chunk_size = int(sample_rate * chunk_duration / 1000)
        self.min_speech_chunks = int(min_speech_duration / chunk_duration)
        self.silence_timeout_chunks = int(silence_timeout / chunk_duration)
        
        # Initialize components using lazy imports
        VoiceDetector = _import_vad()
        self.voice_detector = VoiceDetector(
            aggressiveness=vad_aggressiveness,
            sample_rate=sample_rate,
            debug_mode=debug_mode
        )

        # STT: use faster-whisper adapter by default (core dependency)
        STTAdapter = _import_transcriber()
        self.stt_adapter = STTAdapter(model_size=whisper_model, device="cpu", compute_type="int8")
        self.min_transcription_length = min_transcription_length
        
        # State
        self.is_running = False
        self.thread = None
        self.stream = None
        self.tts_interrupt_callback = None
        self.tts_interrupt_enabled = True  # Can be disabled during TTS playback
        self.listening_paused = False  # Can be paused to completely stop processing audio
    
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

    def _transcribe_pcm16(self, pcm16_bytes: bytes, language: Optional[str] = None) -> str:
        """Transcribe raw PCM16 mono audio bytes."""
        if not pcm16_bytes:
            return ""

        audio = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        text = self.stt_adapter.transcribe_from_array(audio, sample_rate=self.sample_rate, language=language)
        return (text or "").strip()

    def _is_stop_command(self, text: str) -> bool:
        """Return True if text matches a configured stop phrase."""
        if not text:
            return False

        normalized = re.sub(r"[^a-z0-9\s]+", " ", text.lower()).strip()
        normalized = re.sub(r"\s+", " ", normalized)

        # Exact match for configured stop phrases
        if normalized in self.stop_phrases:
            return True

        # Backward compatibility / convenience: allow plain "stop" as well
        if normalized == "stop":
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
                                    # Normal transcription
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