"""Main Voice Manager class for coordinating TTS and STT components."""

from .tts import TTSEngine
from .recognition import VoiceRecognizer


class VoiceManager:
    """Main class for voice interaction capabilities."""
    
    def __init__(self, tts_model="tts_models/en/ljspeech/vits",
                 whisper_model="tiny", debug_mode=False):
        """Initialize the Voice Manager.
        
        Args:
            tts_model: TTS model name to use
            whisper_model: Whisper model name to use
            debug_mode: Enable debug logging
        """
        self.debug_mode = debug_mode
        self.speed = 1.0  
        
        # Initialize TTS engine
        self.tts_engine = TTSEngine(
            model_name=tts_model,
            debug_mode=debug_mode
        )
        
        # Set up callbacks to pause/resume voice recognition during TTS playback
        # This prevents the system from interrupting its own speech
        self.tts_engine.on_playback_start = self._on_tts_start
        self.tts_engine.on_playback_end = self._on_tts_end
        
        # Voice recognizer is initialized on demand
        self.voice_recognizer = None
        self.whisper_model = whisper_model
        
        # State tracking
        self._transcription_callback = None
        self._stop_callback = None
        self._voice_mode = "full"  # full, wait, stop, ptt
    
    def _on_tts_start(self):
        """Called when TTS playback starts - handle based on voice mode."""
        if not self.voice_recognizer:
            return
        
        if self._voice_mode == "full":
            # Full mode: Keep listening but pause interrupt capability
            self.voice_recognizer.pause_tts_interrupt()
        elif self._voice_mode in ["wait", "stop", "ptt"]:
            # Wait/Stop/PTT modes: Pause listening entirely during TTS
            self.voice_recognizer.pause_listening()
    
    def _on_tts_end(self):
        """Called when TTS playback ends - handle based on voice mode."""
        if not self.voice_recognizer:
            return
        
        if self._voice_mode == "full":
            # Full mode: Resume interrupt capability
            self.voice_recognizer.resume_tts_interrupt()
        elif self._voice_mode in ["wait", "stop", "ptt"]:
            # Wait/Stop/PTT modes: Resume listening
            self.voice_recognizer.resume_listening()
    
    def speak(self, text, speed=1.0, callback=None):
        """Convert text to speech and play audio.
        
        Args:
            text: Text to convert to speech
            speed: Speech speed (0.5-2.0)
            callback: Function to call when speech completes
            
        Returns:
            True if speech started, False otherwise
        """
        sp = 1.0
        if speed != 1.0:
            sp = speed
        else:
            sp = self.speed
        
        return self.tts_engine.speak(text, sp, callback)
    
    def stop_speaking(self):
        """Stop current speech playback.
        
        Returns:
            True if stopped, False if no playback was active
        """
        return self.tts_engine.stop()
    
    def pause_speaking(self):
        """Pause current speech playback.
        
        Pauses at chunk boundaries in streaming mode. Can be resumed with resume_speaking().
        
        Returns:
            True if paused, False if no playback was active
        """
        return self.tts_engine.pause()
    
    def resume_speaking(self):
        """Resume paused speech playback.
        
        Returns:
            True if resumed, False if not paused or no playback active
        """
        return self.tts_engine.resume()
    
    def is_paused(self):
        """Check if TTS is currently paused.
        
        Returns:
            True if paused, False otherwise
        """
        return self.tts_engine.is_paused()
    
    def is_speaking(self):
        """Check if TTS is currently active.
        
        Returns:
            True if speaking, False otherwise
        """
        return self.tts_engine.is_active()
    
    def listen(self, on_transcription, on_stop=None):
        """Start listening for speech with callbacks.
        
        Args:
            on_transcription: Callback for transcribed text
            on_stop: Callback when 'stop' command detected
            
        Returns:
            True if started, False if already listening
        """
        # Store callbacks
        self._transcription_callback = on_transcription
        self._stop_callback = on_stop
        
        # Initialize recognizer if not already done
        if not self.voice_recognizer:
            def _transcription_handler(text):
                if self._transcription_callback:
                    self._transcription_callback(text)
            
            def _stop_handler():
                # Stop listening
                self.stop_listening()
                # Call user's stop callback if provided
                if self._stop_callback:
                    self._stop_callback()
            
            self.voice_recognizer = VoiceRecognizer(
                transcription_callback=_transcription_handler,
                stop_callback=_stop_handler,
                whisper_model=self.whisper_model,
                debug_mode=self.debug_mode
            )
        
        # Start with TTS interrupt capability
        return self.voice_recognizer.start(
            tts_interrupt_callback=self.stop_speaking
        )
    
    def stop_listening(self):
        """Stop listening for speech.
        
        Returns:
            True if stopped, False if not listening
        """
        if self.voice_recognizer:
            return self.voice_recognizer.stop()
        return False
    
    def is_listening(self):
        """Check if currently listening for speech.
        
        Returns:
            True if listening, False otherwise
        """
        return self.voice_recognizer and self.voice_recognizer.is_running
    
    def set_voice_mode(self, mode):
        """Set the voice mode (full, wait, stop, ptt).
        
        Args:
            mode: Voice mode to use
            
        Returns:
            True if successful
        """
        if mode in ["full", "wait", "stop", "ptt"]:
            self._voice_mode = mode
            return True
        return False
        
    def set_speed(self, speed):
        """Set the TTS speed.
        
        Args:
            speed: Speech speed multiplier (0.5-2.0)
            
        Returns:
            True if successful
        """
        self.speed = speed
        return True
    
    def get_speed(self):
        """Get the TTS speed.
        
        Returns:
            Current TTS speed multiplier
        """
        return self.speed

    def set_tts_model(self, model_name):
        """Change the TTS model.
        
        Available models (all pure Python, cross-platform):
        - "tts_models/en/ljspeech/fast_pitch" (default, recommended)
        - "tts_models/en/ljspeech/glow-tts" (alternative)
        - "tts_models/en/ljspeech/tacotron2-DDC" (legacy)
        
        Args:
            model_name: TTS model name to use
            
        Returns:
            True if successful
            
        Example:
            vm.set_tts_model("tts_models/en/ljspeech/glow-tts")
        """
        # Stop any current speech
        self.stop_speaking()
        
        # Reinitialize TTS engine with new model
        self.tts_engine = TTSEngine(
            model_name=model_name,
            debug_mode=self.debug_mode
        )
        
        # Restore callbacks
        self.tts_engine.on_playback_start = self._on_tts_start
        self.tts_engine.on_playback_end = self._on_tts_end
        
        return True
    
    def set_whisper(self, model_name):
        """Set the Whisper model.
        
        Args:
            whisper_model: Whisper model name (tiny, base, etc.)
            
        Returns:
            True if successful
        """
        self.whisper_model = model_name
        if self.voice_recognizer:
            return self.voice_recognizer.change_whisper_model(model_name)
    
    def get_whisper(self):
        """Get the Whisper model.
        
        Returns:
            Current Whisper model name
        """
        return self.whisper_model
    
    def change_vad_aggressiveness(self, aggressiveness):
        """Change VAD aggressiveness.
        
        Args:
            aggressiveness: New aggressiveness level (0-3)
            
        Returns:
            True if changed, False otherwise
        """
        if self.voice_recognizer:
            return self.voice_recognizer.change_vad_aggressiveness(aggressiveness)
        return False
    
    def cleanup(self):
        """Clean up resources.
        
        Returns:
            True if cleanup successful
        """
        if self.voice_recognizer:
            self.voice_recognizer.stop()
        
        self.stop_speaking()
        return True 