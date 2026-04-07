"""Base adapter interfaces for TTS and STT engines.

These abstract base classes define the contract that all TTS and STT adapters
must implement, ensuring consistent API across different backends.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Union
import numpy as np
import io

from ..voice_profiles import VoiceProfile


class TTSAdapter(ABC):
    """Abstract base class for Text-to-Speech adapters.
    
    All TTS engines must implement this interface to be compatible with
    the VoiceManager. This ensures we can swap engines without breaking
    existing code.
    """
    
    @abstractmethod
    def synthesize(self, text: str) -> np.ndarray:
        """Convert text to audio array for immediate playback.
        
        Args:
            text: The text to synthesize
            
        Returns:
            Audio data as numpy array (shape: [samples,], dtype: float32, range: -1.0 to 1.0)
        """
        pass
    
    @abstractmethod
    def synthesize_to_bytes(self, text: str, format: str = 'wav') -> bytes:
        """Convert text to audio bytes for network transmission or file storage.
        
        This method is essential for client-server architectures where the backend
        generates speech and sends it to clients for playback.
        
        Args:
            text: The text to synthesize
            format: Audio format ('wav', 'mp3', 'ogg'). Default: 'wav'
            
        Returns:
            Audio data as bytes in the specified format
        """
        pass
    
    @abstractmethod
    def synthesize_to_file(self, text: str, output_path: str, format: Optional[str] = None) -> str:
        """Convert text to audio file.
        
        Args:
            text: The text to synthesize
            output_path: Path to save the audio file
            format: Audio format (optional, inferred from file extension if not provided)
            
        Returns:
            Path to the saved audio file
        """
        pass
    
    @abstractmethod
    def set_language(self, language: str) -> bool:
        """Switch the TTS language.
        
        Args:
            language: ISO 639-1 language code (e.g., 'en', 'fr', 'de')
            
        Returns:
            True if language switch successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_supported_languages(self) -> list[str]:
        """Get list of supported language codes.
        
        Returns:
            List of ISO 639-1 language codes
        """
        pass
    
    @abstractmethod
    def get_sample_rate(self) -> int:
        """Get the sample rate of the synthesized audio.
        
        Returns:
            Sample rate in Hz (e.g., 22050, 16000)
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this TTS engine is available and functional.
        
        Returns:
            True if the engine can be used, False if dependencies missing or initialization failed
        """
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Get metadata about this TTS engine.
        
        Returns:
            Dictionary with engine information (name, version, languages, etc.)
        """
        return {
            'name': self.__class__.__name__,
            'languages': self.get_supported_languages(),
            'sample_rate': self.get_sample_rate(),
            'available': self.is_available()
        }

    # ---------------------------------------------------------------------
    # Optional voice profile interface (engine-agnostic presets)
    # ---------------------------------------------------------------------

    def get_profiles(self) -> list[VoiceProfile]:
        """Return available voice profiles for this adapter (best-effort).

        Engines without a profile notion should return an empty list.
        """
        return []

    def set_profile(self, profile_id: str) -> bool:
        """Apply a voice profile by id (best-effort).

        Engines without profiles should return False.
        """
        _ = profile_id
        return False

    def get_active_profile(self) -> VoiceProfile | None:
        """Return the active profile, if known (best-effort)."""
        return None

    # ---------------------------------------------------------------------
    # Optional quality preset (engine-agnostic knob)
    # ---------------------------------------------------------------------

    def set_quality_preset(self, preset: str) -> bool:
        """Best-effort speed/quality preset for this TTS engine.

        Presets are intended to be engine-agnostic (`low|standard|high`).
        Backward-compatible aliases may exist (`fast`→`low`, `balanced`→`standard`).
        Engines that do not support quality tuning may return False.
        """
        _ = preset
        return False

    def get_quality_preset(self) -> Optional[str]:
        """Return the currently active quality preset, if supported."""
        return None


class STTAdapter(ABC):
    """Abstract base class for Speech-to-Text adapters.
    
    All STT engines must implement this interface to be compatible with
    the VoiceManager.
    """
    
    @abstractmethod
    def transcribe(self, audio_path: str, language: Optional[str] = None) -> str:
        """Transcribe audio file to text.
        
        Args:
            audio_path: Path to audio file
            language: Target language (optional, auto-detect if not provided)
            
        Returns:
            Transcribed text
        """
        pass
    
    @abstractmethod
    def transcribe_from_bytes(self, audio_bytes: bytes, language: Optional[str] = None) -> str:
        """Transcribe audio from bytes (network use case).
        
        This method is essential for client-server architectures where clients
        record audio and send it to the backend for transcription.
        
        Args:
            audio_bytes: Audio data as bytes
            language: Target language (optional, auto-detect if not provided)
            
        Returns:
            Transcribed text
        """
        pass
    
    @abstractmethod
    def transcribe_from_array(self, audio_array: np.ndarray, sample_rate: int, 
                             language: Optional[str] = None) -> str:
        """Transcribe audio from numpy array.
        
        Args:
            audio_array: Audio data as numpy array
            sample_rate: Sample rate of the audio in Hz
            language: Target language (optional, auto-detect if not provided)
            
        Returns:
            Transcribed text
        """
        pass
    
    @abstractmethod
    def set_language(self, language: str) -> bool:
        """Set the default language for transcription.
        
        Args:
            language: ISO 639-1 language code
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_supported_languages(self) -> list[str]:
        """Get list of supported language codes.
        
        Returns:
            List of ISO 639-1 language codes
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this STT engine is available and functional.
        
        Returns:
            True if the engine can be used, False otherwise
        """
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Get metadata about this STT engine.
        
        Returns:
            Dictionary with engine information
        """
        return {
            'name': self.__class__.__name__,
            'languages': self.get_supported_languages(),
            'available': self.is_available()
        }
