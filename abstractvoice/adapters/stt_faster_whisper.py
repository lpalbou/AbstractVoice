"""Faster-Whisper STT Adapter - High-performance speech recognition.

Faster-Whisper is a reimplementation of OpenAI's Whisper using CTranslate2:
- 4x faster inference than openai-whisper
- 60% lower memory usage with INT8 quantization
- Same accuracy as openai-whisper
- Better CPU performance
- Supports GPU acceleration (CUDA) if available
"""

import os
import io
import logging
import numpy as np
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any
import wave

from .base import STTAdapter

logger = logging.getLogger(__name__)


class FasterWhisperAdapter(STTAdapter):
    """Faster-Whisper STT adapter using faster-whisper package.
    
    This adapter provides high-performance speech-to-text with same accuracy
    as openai-whisper but 4x faster and 60% lower memory usage.
    """
    
    # Supported models (size -> (parameters, speed, accuracy))
    MODELS = {
        'tiny': {'params': '39M', 'speed': 'very_fast', 'accuracy': 'low'},
        'base': {'params': '74M', 'speed': 'fast', 'accuracy': 'good'},  # Default
        'small': {'params': '244M', 'speed': 'medium', 'accuracy': 'better'},
        'medium': {'params': '769M', 'speed': 'slow', 'accuracy': 'high'},
        'large-v2': {'params': '1550M', 'speed': 'very_slow', 'accuracy': 'best'},
        'large-v3': {'params': '1550M', 'speed': 'very_slow', 'accuracy': 'best'},
    }
    
    # Supported languages
    LANGUAGES = [
        'en', 'fr', 'de', 'es', 'ru', 'zh',  # Required 6
        'it', 'pt', 'ja', 'ko', 'ar', 'hi',  # Additional common languages
    ]
    
    def __init__(self, model_size: str = 'base', device: str = 'cpu', compute_type: str = 'int8'):
        """Initialize Faster-Whisper STT adapter.
        
        Args:
            model_size: Model size ('tiny', 'base', 'small', 'medium', 'large-v2', 'large-v3')
            device: Device to run on ('cpu', 'cuda', 'auto')
            compute_type: Computation type ('int8', 'float16', 'float32')
                         int8 provides 60% memory reduction with minimal accuracy loss
        """
        self._faster_whisper_available = False
        self._model = None
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._current_language = None
        
        # Try to import faster-whisper
        try:
            from faster_whisper import WhisperModel
            self._WhisperModel = WhisperModel
            self._faster_whisper_available = True
            logger.info("✅ Faster-Whisper initialized successfully")
            
            # Load model
            self._load_model(model_size, device, compute_type)
            
        except ImportError as e:
            logger.warning(f"⚠️  Faster-Whisper not available: {e}")
            logger.info(
                "To install Faster-Whisper:\n"
                "  pip install faster-whisper>=0.10.0\n"
                "This will enable 4x faster STT with same accuracy."
            )
    
    def _load_model(self, model_size: str, device: str = 'cpu', compute_type: str = 'int8') -> bool:
        """Load Faster-Whisper model.
        
        Args:
            model_size: Model size
            device: Device ('cpu', 'cuda', 'auto')
            compute_type: Computation type ('int8', 'float16', 'float32')
            
        Returns:
            True if successful, False otherwise
        """
        if not self._faster_whisper_available:
            return False
        
        if model_size not in self.MODELS:
            logger.warning(f"⚠️  Unknown model size '{model_size}', using 'base'")
            model_size = 'base'
        
        try:
            logger.info(f"⬇️  Loading Faster-Whisper model: {model_size} ({self.MODELS[model_size]['params']})")
            
            # Load model (will auto-download if not cached)
            self._model = self._WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
                download_root=None  # Use default cache (~/.cache/huggingface)
            )
            
            self._model_size = model_size
            self._device = device
            self._compute_type = compute_type
            
            logger.info(f"✅ Loaded Faster-Whisper model: {model_size}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load Faster-Whisper model: {e}")
            return False
    
    def transcribe(self, audio_path: str, language: Optional[str] = None) -> str:
        """Transcribe audio file to text.
        
        Args:
            audio_path: Path to audio file
            language: Target language (optional, auto-detect if not provided)
            
        Returns:
            Transcribed text
        """
        if not self.is_available():
            raise RuntimeError(
                "Faster-Whisper is not available. Install with: pip install faster-whisper>=0.10.0"
            )
        
        try:
            # Transcribe with faster-whisper
            segments, info = self._model.transcribe(
                audio_path,
                language=language,
                beam_size=5,
                best_of=5,
                temperature=0.0,
                vad_filter=True,  # Use Voice Activity Detection
                vad_parameters=dict(min_silence_duration_ms=500),
            )
            
            # Combine all segments
            text = " ".join([segment.text.strip() for segment in segments])
            
            if language is None:
                logger.debug(f"Detected language: {info.language} (confidence: {info.language_probability:.2f})")
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"❌ Faster-Whisper transcription failed: {e}")
            raise RuntimeError(f"Transcription failed: {e}") from e
    
    def transcribe_from_bytes(self, audio_bytes: bytes, language: Optional[str] = None) -> str:
        """Transcribe audio from bytes (network use case).
        
        Args:
            audio_bytes: Audio data as bytes (WAV format)
            language: Target language (optional, auto-detect if not provided)
            
        Returns:
            Transcribed text
        """
        # Save bytes to temporary file and transcribe
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name
        
        try:
            return self.transcribe(tmp_path, language=language)
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except:
                pass
    
    def transcribe_from_array(self, audio_array: np.ndarray, sample_rate: int,
                             language: Optional[str] = None) -> str:
        """Transcribe audio from numpy array.
        
        Args:
            audio_array: Audio data as numpy array (float32, range -1.0 to 1.0)
            sample_rate: Sample rate of the audio in Hz
            language: Target language (optional, auto-detect if not provided)
            
        Returns:
            Transcribed text
        """
        # Convert array to WAV bytes
        audio_bytes = self._array_to_wav_bytes(audio_array, sample_rate)
        
        # Transcribe from bytes
        return self.transcribe_from_bytes(audio_bytes, language=language)
    
    def _array_to_wav_bytes(self, audio_array: np.ndarray, sample_rate: int) -> bytes:
        """Convert numpy array to WAV bytes.
        
        Args:
            audio_array: Audio as float32 array [-1.0, 1.0]
            sample_rate: Sample rate in Hz
            
        Returns:
            WAV file as bytes
        """
        # Convert to 16-bit PCM
        audio_int16 = (audio_array * 32767).astype(np.int16)
        
        # Create WAV file in memory
        buffer = io.BytesIO()
        
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        
        return buffer.getvalue()
    
    def set_language(self, language: str) -> bool:
        """Set the default language for transcription.
        
        Args:
            language: ISO 639-1 language code
            
        Returns:
            True if successful, False otherwise
        """
        if language not in self.LANGUAGES:
            logger.warning(f"⚠️  Language {language} may not be well-supported")
            return False
        
        self._current_language = language
        return True
    
    def get_supported_languages(self) -> list[str]:
        """Get list of supported language codes.
        
        Returns:
            List of ISO 639-1 language codes
        """
        return self.LANGUAGES.copy()
    
    def is_available(self) -> bool:
        """Check if Faster-Whisper is available and functional.
        
        Returns:
            True if the engine can be used, False otherwise
        """
        return self._faster_whisper_available and self._model is not None
    
    def change_model(self, model_size: str) -> bool:
        """Change the Whisper model size.
        
        Args:
            model_size: New model size ('tiny', 'base', 'small', 'medium', 'large-v2', 'large-v3')
            
        Returns:
            True if successful, False otherwise
        """
        if model_size == self._model_size:
            logger.debug(f"Model {model_size} already loaded")
            return True
        
        return self._load_model(model_size, self._device, self._compute_type)
    
    def get_info(self) -> Dict[str, Any]:
        """Get metadata about Faster-Whisper engine.
        
        Returns:
            Dictionary with engine information
        """
        info = super().get_info()
        info.update({
            'engine': 'Faster-Whisper',
            'version': '1.2.0+',
            'model_size': self._model_size,
            'model_params': self.MODELS.get(self._model_size, {}).get('params', 'unknown'),
            'device': self._device,
            'compute_type': self._compute_type,
            'current_language': self._current_language,
            'performance': f"{self.MODELS.get(self._model_size, {}).get('speed', 'unknown')} speed, "
                          f"{self.MODELS.get(self._model_size, {}).get('accuracy', 'unknown')} accuracy",
            'memory_optimization': 'INT8 quantization' if self._compute_type == 'int8' else None
        })
        return info
