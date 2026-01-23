"""Piper TTS Adapter - Zero-dependency TTS engine.

Piper is a fast, local neural text-to-speech system that:
- Requires NO system dependencies (no espeak-ng)
- Works on Windows, macOS, Linux out of the box
- Supports 40+ languages with 100+ voices
- Uses ONNX Runtime for cross-platform compatibility
- Has small model sizes (15-60MB vs 200-500MB VITS)
"""

import os
import io
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import wave
import struct

from .base import TTSAdapter

logger = logging.getLogger(__name__)


class PiperTTSAdapter(TTSAdapter):
    """Piper TTS adapter using piper-tts package.
    
    This adapter provides cross-platform TTS without system dependencies,
    making it ideal for easy installation on Windows, macOS, and Linux.
    """
    
    # Language-to-voice mapping (using quality 'medium' models for balance of size/quality)
    # Format: language_code -> (hf_path, model_filename)
    PIPER_MODELS = {
        'en': ('en/en_US/amy/medium', 'en_US-amy-medium'),         # US English, female voice
        'fr': ('fr/fr_FR/siwis/medium', 'fr_FR-siwis-medium'),     # France French
        'de': ('de/de_DE/thorsten/medium', 'de_DE-thorsten-medium'), # German
        'es': ('es/es_ES/carlfm/medium', 'es_ES-carlfm-medium'),   # Spain Spanish
        'ru': ('ru/ru_RU/dmitri/medium', 'ru_RU-dmitri-medium'),   # Russian
        'zh': ('zh/zh_CN/huayan/medium', 'zh_CN-huayan-medium'),   # Mandarin Chinese
    }
    
    # Model download sizes (for user information)
    MODEL_SIZES = {
        'en': '50MB',
        'fr': '45MB',
        'de': '48MB',
        'es': '47MB',
        'ru': '52MB',
        'zh': '55MB',
    }
    
    def __init__(self, language: str = 'en', model_dir: Optional[str] = None):
        """Initialize Piper TTS adapter.
        
        Args:
            language: Initial language (default: 'en')
            model_dir: Directory to store models (default: ~/.piper/models)
        """
        self._piper_available = False
        self._voice = None
        self._current_language = None
        self._sample_rate = 22050  # Piper default
        
        # Set model directory
        if model_dir is None:
            home = Path.home()
            self._model_dir = home / '.piper' / 'models'
        else:
            self._model_dir = Path(model_dir)
        
        self._model_dir.mkdir(parents=True, exist_ok=True)
        
        # Try to import piper-tts
        try:
            from piper import PiperVoice
            self._PiperVoice = PiperVoice
            self._piper_available = True
            logger.info("✅ Piper TTS initialized successfully")
            
            # Load initial language model
            self.set_language(language)
            
        except ImportError as e:
            logger.warning(f"⚠️  Piper TTS not available: {e}")
            logger.info(
                "To install Piper TTS:\n"
                "  pip install piper-tts>=1.2.0\n"
                "This will enable zero-dependency TTS on all platforms."
            )
    
    def _get_model_path(self, language: str) -> tuple[Path, Path]:
        """Get paths for model and config files.
        
        Args:
            language: Language code
            
        Returns:
            Tuple of (model_path, config_path)
        """
        model_info = self.PIPER_MODELS.get(language)
        if not model_info:
            raise ValueError(f"Unsupported language: {language}")
        
        _, model_filename = model_info
        model_path = self._model_dir / f"{model_filename}.onnx"
        config_path = self._model_dir / f"{model_filename}.onnx.json"
        
        return model_path, config_path
    
    def _download_model(self, language: str) -> bool:
        """Download Piper model for specified language using Hugging Face Hub.
        
        Args:
            language: Language code
            
        Returns:
            True if successful, False otherwise
        """
        if not self._piper_available:
            return False
        
        model_info = self.PIPER_MODELS.get(language)
        if not model_info:
            logger.error(f"❌ No Piper model defined for language: {language}")
            return False
        
        hf_path, model_filename = model_info
        model_path, config_path = self._get_model_path(language)
        
        # Check if already downloaded
        if model_path.exists() and config_path.exists():
            logger.debug(f"✅ Model already exists: {model_filename}")
            return True
        
        # Download from Piper repository.
        #
        # IMPORTANT: we intentionally avoid importing `huggingface_hub` here.
        # In some environments we've observed intermittent interpreter crashes
        # during deep import chains (pure-Python packages should not segfault,
        # which strongly suggests native extension interactions elsewhere).
        #
        # Using direct HTTPS downloads is simpler, more predictable, and keeps
        # the adapter robust in "fresh install" scenarios.
        logger.info(f"⬇️  Downloading Piper model for {language} ({self.MODEL_SIZES.get(language, 'unknown size')})...")
        
        try:
            repo_id = "rhasspy/piper-voices"
            base_url = f"https://huggingface.co/{repo_id}/resolve/main"

            def _download(url: str, dest: Path) -> None:
                import requests
                import tempfile

                dest.parent.mkdir(parents=True, exist_ok=True)

                with requests.get(url, stream=True, timeout=60) as r:
                    r.raise_for_status()

                    # Write atomically to avoid leaving corrupt partial files.
                    with tempfile.NamedTemporaryFile(dir=str(dest.parent), delete=False) as tmp:
                        for chunk in r.iter_content(chunk_size=1024 * 256):
                            if chunk:
                                tmp.write(chunk)
                        tmp_path = Path(tmp.name)

                tmp_path.replace(dest)

            # Download model file
            if not model_path.exists():
                logger.info(f"   Downloading {model_path.name}...")
                _download(f"{base_url}/{hf_path}/{model_filename}.onnx", model_path)

            # Download config file
            if not config_path.exists():
                logger.info(f"   Downloading {config_path.name}...")
                _download(f"{base_url}/{hf_path}/{model_filename}.onnx.json", config_path)
            
            logger.info(f"✅ Successfully downloaded Piper model for {language}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to download Piper model: {e}")
            logger.info(f"   If this persists, manually download from: https://huggingface.co/rhasspy/piper-voices")
            # Clean up partial downloads
            if model_path.exists():
                model_path.unlink()
            if config_path.exists():
                config_path.unlink()
            return False
    
    def _load_voice(self, language: str) -> bool:
        """Load Piper voice for specified language.
        
        Args:
            language: Language code
            
        Returns:
            True if successful, False otherwise
        """
        if not self._piper_available:
            return False
        
        # Download model if needed
        model_path, config_path = self._get_model_path(language)
        if not (model_path.exists() and config_path.exists()):
            if not self._download_model(language):
                return False
        
        # Load the voice
        try:
            logger.debug(f"Loading Piper voice: {model_path}")
            self._voice = self._PiperVoice.load(str(model_path), str(config_path))
            self._current_language = language
            
            # Update sample rate from config
            if hasattr(self._voice, 'config') and hasattr(self._voice.config, 'sample_rate'):
                self._sample_rate = self._voice.config.sample_rate
            
            logger.info(f"✅ Loaded Piper voice for {language}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load Piper voice for {language}: {e}")
            return False
    
    def synthesize(self, text: str) -> np.ndarray:
        """Convert text to audio array for immediate playback.
        
        Args:
            text: The text to synthesize
            
        Returns:
            Audio data as numpy array (float32, range -1.0 to 1.0)
        """
        if not self.is_available():
            raise RuntimeError("Piper TTS is not available. Install with: pip install piper-tts>=1.2.0")
        
        if not self._voice:
            raise RuntimeError(f"No voice loaded. Call set_language() first.")
        
        try:
            # Piper synthesize returns an iterable of AudioChunk objects
            audio_chunks = list(self._voice.synthesize(text))
            
            if not audio_chunks:
                return np.array([], dtype=np.float32)
            
            # Combine all audio chunks into single array
            # Each chunk has audio_float_array attribute with normalized float32 audio
            audio_arrays = [chunk.audio_float_array for chunk in audio_chunks]
            
            # Concatenate all arrays
            audio_array = np.concatenate(audio_arrays)
            
            return audio_array
            
        except Exception as e:
            logger.error(f"❌ Piper synthesis failed: {e}")
            raise RuntimeError(f"Piper synthesis failed: {e}") from e
    
    def synthesize_to_bytes(self, text: str, format: str = 'wav') -> bytes:
        """Convert text to audio bytes for network transmission.
        
        Args:
            text: The text to synthesize
            format: Audio format ('wav' only supported currently)
            
        Returns:
            Audio data as bytes in WAV format
        """
        if format.lower() != 'wav':
            raise ValueError(f"Piper adapter currently only supports WAV format, not {format}")
        
        # Get audio array
        audio_array = self.synthesize(text)
        
        # Convert to bytes
        return self._array_to_wav_bytes(audio_array)
    
    def synthesize_to_file(self, text: str, output_path: str, format: Optional[str] = None) -> str:
        """Convert text to audio file.
        
        Args:
            text: The text to synthesize
            output_path: Path to save the audio file
            format: Audio format (optional, inferred from extension)
            
        Returns:
            Path to the saved audio file
        """
        # Infer format from extension if not provided
        if format is None:
            format = Path(output_path).suffix.lstrip('.')
        
        if format.lower() != 'wav':
            raise ValueError(f"Piper adapter currently only supports WAV format, not {format}")
        
        # Get audio bytes
        audio_bytes = self.synthesize_to_bytes(text, format='wav')
        
        # Write to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            f.write(audio_bytes)
        
        logger.info(f"✅ Saved audio to: {output_path}")
        return str(output_path)
    
    def _array_to_wav_bytes(self, audio_array: np.ndarray) -> bytes:
        """Convert numpy array to WAV bytes.
        
        Args:
            audio_array: Audio as float32 array [-1.0, 1.0]
            
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
            wav_file.setframerate(self._sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        
        return buffer.getvalue()
    
    def set_language(self, language: str) -> bool:
        """Switch the TTS language.
        
        Args:
            language: ISO 639-1 language code (e.g., 'en', 'fr', 'de')
            
        Returns:
            True if language switch successful, False otherwise
        """
        if language not in self.PIPER_MODELS:
            logger.warning(f"⚠️  Language {language} not supported by Piper adapter")
            return False
        
        # Don't reload if already loaded
        if self._current_language == language and self._voice is not None:
            logger.debug(f"Language {language} already loaded")
            return True
        
        # Load new voice
        return self._load_voice(language)
    
    def get_supported_languages(self) -> list[str]:
        """Get list of supported language codes.
        
        Returns:
            List of ISO 639-1 language codes
        """
        return list(self.PIPER_MODELS.keys())
    
    def get_sample_rate(self) -> int:
        """Get the sample rate of the synthesized audio.
        
        Returns:
            Sample rate in Hz (typically 22050)
        """
        return self._sample_rate
    
    def is_available(self) -> bool:
        """Check if Piper TTS is available and functional.
        
        Returns:
            True if Piper can be used, False otherwise
        """
        return self._piper_available and self._voice is not None
    
    def get_info(self) -> Dict[str, Any]:
        """Get metadata about Piper TTS engine.
        
        Returns:
            Dictionary with engine information
        """
        info = super().get_info()
        info.update({
            'engine': 'Piper TTS',
            'version': '1.2.0+',
            'current_language': self._current_language,
            'model_dir': str(self._model_dir),
            'requires_system_deps': False,
            'cross_platform': True
        })
        return info
