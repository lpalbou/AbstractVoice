"""Voice recognition module that combines VAD and STT."""

import threading
import time
import pyaudio
from .vad import VoiceDetector
from .stt import Transcriber


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
        
        # Configuration
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration  # in ms
        self.chunk_size = int(sample_rate * chunk_duration / 1000)
        self.min_speech_chunks = int(min_speech_duration / chunk_duration)
        self.silence_timeout_chunks = int(silence_timeout / chunk_duration)
        
        # Initialize components
        self.voice_detector = VoiceDetector(
            aggressiveness=vad_aggressiveness,
            sample_rate=sample_rate,
            debug_mode=debug_mode
        )
        
        self.transcriber = Transcriber(
            model_name=whisper_model,
            min_transcription_length=min_transcription_length,
            debug_mode=debug_mode
        )
        
        # State
        self.is_running = False
        self.thread = None
        self.pyaudio = None
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
            self.stream.stop_stream()
            self.stream.close()
        
        if self.pyaudio:
            self.pyaudio.terminate()
        
        if self.debug_mode:
            print(" > Voice recognition stopped")
        return True
    
    def _recognition_loop(self):
        """Main recognition loop."""
        import pyaudio
        
        self.pyaudio = pyaudio.PyAudio()
        self.stream = self.pyaudio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        
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
                audio_data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                
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
                            text = self.transcriber.transcribe(audio_bytes)
                            
                            if text:
                                # Check for stop command
                                if text.lower() == "stop":
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
        return self.transcriber.change_model(model_name)
    
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