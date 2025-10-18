"""Voice activity detection using WebRTC VAD."""

import webrtcvad
import logging


class VoiceDetector:
    """Detects voice activity in audio streams."""
    
    def __init__(self, aggressiveness=1, sample_rate=16000, debug_mode=False):
        """Initialize the voice detector.
        
        Args:
            aggressiveness: VAD aggressiveness (0-3, higher is more strict)
            sample_rate: Audio sample rate (8000, 16000, 32000, 48000 Hz)
            debug_mode: Enable debug output
        """
        self.debug_mode = debug_mode
        self.sample_rate = sample_rate
        self.aggressiveness = aggressiveness
        
        # Check sample rate is valid for WebRTC VAD
        if sample_rate not in [8000, 16000, 32000, 48000]:
            raise ValueError("Sample rate must be 8000, 16000, 32000, or 48000 Hz")
        
        # Initialize WebRTC VAD
        try:
            self.vad = webrtcvad.Vad(aggressiveness)
            if self.debug_mode:
                print(f" > VAD initialized with aggressiveness {aggressiveness}")
        except Exception as e:
            if self.debug_mode:
                print(f"VAD initialization error: {e}")
            raise
    
    def is_speech(self, audio_frame):
        """Check if audio frame contains speech.
        
        Args:
            audio_frame: Audio frame as bytes (must be 10, 20, or 30ms at sample_rate)
            
        Returns:
            True if speech detected, False otherwise
        """
        try:
            return self.vad.is_speech(audio_frame, self.sample_rate)
        except Exception as e:
            if self.debug_mode:
                print(f"VAD processing error: {e}")
            return False
    
    def set_aggressiveness(self, aggressiveness):
        """Change VAD aggressiveness.
        
        Args:
            aggressiveness: New aggressiveness level (0-3)
            
        Returns:
            True if changed, False otherwise
        """
        if 0 <= aggressiveness <= 3:
            try:
                self.vad.set_mode(aggressiveness)
                self.aggressiveness = aggressiveness
                if self.debug_mode:
                    print(f" > VAD aggressiveness changed to {aggressiveness}")
                return True
            except Exception as e:
                if self.debug_mode:
                    print(f"VAD aggressiveness change error: {e}")
                return False
        else:
            if self.debug_mode:
                print(f" > Invalid aggressiveness: {aggressiveness}")
            return False 