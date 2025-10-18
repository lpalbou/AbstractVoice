"""Speech-to-text transcription using OpenAI's Whisper."""

import whisper
import numpy as np
import os
import sys
import logging


class Transcriber:
    """Transcribes audio using OpenAI's Whisper model."""
    
    def __init__(self, model_name="tiny", min_transcription_length=5, debug_mode=False):
        """Initialize the Whisper transcriber.
        
        Args:
            model_name: Whisper model to use (tiny, base, etc.)
            min_transcription_length: Minimum length of text to consider valid
            debug_mode: Enable debug output
        """
        self.model_name = model_name
        self.min_transcription_length = min_transcription_length
        self.debug_mode = debug_mode
        
        # Suppress Whisper output unless in debug mode
        if not debug_mode:
            logging.getLogger('whisper').setLevel(logging.ERROR)
            
        try:
            if self.debug_mode:
                print(f" > Loading Whisper model: {model_name}")
                
            # Redirect stdout when loading Whisper model in non-debug mode
            orig_stdout = None
            null_out = None
            if not debug_mode:
                orig_stdout = sys.stdout
                null_out = open(os.devnull, 'w')
                sys.stdout = null_out
                
            # Load the Whisper model
            self.model = whisper.load_model(model_name)
        finally:
            # Restore stdout if we redirected it
            if not debug_mode and orig_stdout:
                sys.stdout = orig_stdout
                if 'null_out' in locals() and null_out:
                    null_out.close()
    
    def transcribe(self, audio_data):
        """Transcribe audio data to text.
        
        Args:
            audio_data: Audio data as bytes or numpy array
            
        Returns:
            Transcribed text or None if transcription failed or is too short
        """
        try:
            # Convert bytes to numpy array if needed
            if isinstance(audio_data, bytes):
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            else:
                audio_np = audio_data
            
            # Redirect stdout for non-debug mode
            orig_stdout = None
            null_out = None
            if not self.debug_mode:
                orig_stdout = sys.stdout
                null_out = open(os.devnull, 'w')
                sys.stdout = null_out
                
            try:
                # Perform transcription
                result = self.model.transcribe(audio_np, fp16=False)
            finally:
                # Restore stdout if we redirected it
                if not self.debug_mode and orig_stdout:
                    sys.stdout = orig_stdout
                    if null_out:
                        null_out.close()
            
            # Extract and clean text
            text = result["text"].strip()
            
            # Skip short transcriptions (except "stop" command)
            if len(text) < self.min_transcription_length and text.lower() != "stop":
                return None
            
            if self.debug_mode:
                print(f" > Transcribed: '{text}'")
                
            return text
            
        except Exception as e:
            if self.debug_mode:
                print(f"Transcription error: {e}")
            return None
    
    def change_model(self, model_name):
        """Change the Whisper model.
        
        Args:
            model_name: New model name (tiny, base, etc.)
            
        Returns:
            True if model was changed, False otherwise
        """
        if model_name in ["tiny", "base", "small", "medium", "large"]:
            if self.debug_mode:
                print(f" > Changing Whisper model to {model_name}")
                
            # Redirect stdout for non-debug mode
            orig_stdout = None
            null_out = None
            if not self.debug_mode:
                orig_stdout = sys.stdout
                null_out = open(os.devnull, 'w')
                sys.stdout = null_out
                
            try:
                self.model = whisper.load_model(model_name)
                self.model_name = model_name
            finally:
                # Restore stdout if we redirected it
                if not self.debug_mode and orig_stdout:
                    sys.stdout = orig_stdout
                    if null_out:
                        null_out.close()
                        
            if self.debug_mode:
                print(f" > Whisper model changed to {model_name}")
            return True
        else:
            if self.debug_mode:
                print(f" > Invalid model name: {model_name}")
            return False 