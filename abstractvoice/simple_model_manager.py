"""
Simple Model Manager for AbstractVoice

Provides clean, simple APIs for model management that can be used by both
CLI commands and third-party applications.
"""

import os
import json
import time
import threading
from typing import Dict, List, Optional, Callable, Any
from pathlib import Path


def _import_tts():
    """Import TTS with helpful error message if dependencies missing."""
    try:
        from TTS.api import TTS
        from TTS.utils.manage import ModelManager
        return TTS, ModelManager
    except ImportError as e:
        raise ImportError(
            "TTS functionality requires coqui-tts. Install with:\n"
            "  pip install abstractvoice[tts]\n"
            f"Original error: {e}"
        ) from e


class SimpleModelManager:
    """Simple, clean model manager for AbstractVoice."""

    # Essential model - guaranteed to work everywhere, reasonable size
    ESSENTIAL_MODEL = "tts_models/en/ljspeech/fast_pitch"

    # Available models organized by language with metadata
    AVAILABLE_MODELS = {
        "en": {
            "fast_pitch": {
                "model": "tts_models/en/ljspeech/fast_pitch",
                "name": "Fast Pitch (English)",
                "quality": "good",
                "size_mb": 107,
                "description": "Lightweight, reliable English voice",
                "requires_espeak": False,
                "default": True
            },
            "vits": {
                "model": "tts_models/en/ljspeech/vits",
                "name": "VITS (English)",
                "quality": "excellent",
                "size_mb": 328,
                "description": "High-quality English voice with natural prosody",
                "requires_espeak": True,
                "default": False
            },
            "tacotron2": {
                "model": "tts_models/en/ljspeech/tacotron2-DDC",
                "name": "Tacotron2 (English)",
                "quality": "good",
                "size_mb": 362,
                "description": "Classic English voice, reliable",
                "requires_espeak": False,
                "default": False
            }
        },
        "fr": {
            "css10_vits": {
                "model": "tts_models/fr/css10/vits",
                "name": "CSS10 VITS (French)",
                "quality": "excellent",
                "size_mb": 548,
                "description": "High-quality French voice",
                "requires_espeak": True,
                "default": True
            },
            "mai_tacotron2": {
                "model": "tts_models/fr/mai/tacotron2-DDC",
                "name": "MAI Tacotron2 (French)",
                "quality": "good",
                "size_mb": 362,
                "description": "Reliable French voice",
                "requires_espeak": False,
                "default": False
            }
        },
        "es": {
            "mai_tacotron2": {
                "model": "tts_models/es/mai/tacotron2-DDC",
                "name": "MAI Tacotron2 (Spanish)",
                "quality": "good",
                "size_mb": 362,
                "description": "Reliable Spanish voice",
                "requires_espeak": False,
                "default": True
            },
            "css10_vits": {
                "model": "tts_models/es/css10/vits",
                "name": "CSS10 VITS (Spanish)",
                "quality": "excellent",
                "size_mb": 548,
                "description": "High-quality Spanish voice",
                "requires_espeak": True,
                "default": False
            }
        },
        "de": {
            "thorsten_vits": {
                "model": "tts_models/de/thorsten/vits",
                "name": "Thorsten VITS (German)",
                "quality": "excellent",
                "size_mb": 548,
                "description": "High-quality German voice",
                "requires_espeak": True,
                "default": True
            }
        },
        "it": {
            "mai_male_vits": {
                "model": "tts_models/it/mai_male/vits",
                "name": "MAI Male VITS (Italian)",
                "quality": "excellent",
                "size_mb": 548,
                "description": "High-quality Italian male voice",
                "requires_espeak": True,
                "default": True
            },
            "mai_female_vits": {
                "model": "tts_models/it/mai_female/vits",
                "name": "MAI Female VITS (Italian)",
                "quality": "excellent",
                "size_mb": 548,
                "description": "High-quality Italian female voice",
                "requires_espeak": True,
                "default": False
            }
        }
    }

    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
        self._cache_dir = None

    @property
    def cache_dir(self) -> str:
        """Get the TTS model cache directory."""
        if self._cache_dir is None:
            # Check common cache locations
            import appdirs
            potential_dirs = [
                os.path.expanduser("~/.cache/tts"),
                appdirs.user_data_dir("tts"),
                os.path.expanduser("~/.local/share/tts"),
                os.path.expanduser("~/Library/Application Support/tts"),  # macOS
            ]

            # Find existing cache or use default
            for cache_dir in potential_dirs:
                if os.path.exists(cache_dir):
                    self._cache_dir = cache_dir
                    break
            else:
                # Use appdirs default
                self._cache_dir = appdirs.user_data_dir("tts")

        return self._cache_dir

    def is_model_cached(self, model_name: str) -> bool:
        """Check if a specific model is cached locally."""
        try:
            # Convert model name to cache directory structure
            cache_name = model_name.replace("/", "--")
            model_path = os.path.join(self.cache_dir, cache_name)

            if not os.path.exists(model_path):
                return False

            # Check for essential model files
            essential_files = ["model.pth", "config.json"]
            return any(os.path.exists(os.path.join(model_path, f)) for f in essential_files)
        except Exception as e:
            if self.debug_mode:
                print(f"Error checking cache for {model_name}: {e}")
            return False

    def download_model(self, model_name: str, progress_callback: Optional[Callable[[str, bool], None]] = None) -> bool:
        """Download a specific model.

        Args:
            model_name: TTS model name (e.g., 'tts_models/en/ljspeech/fast_pitch')
            progress_callback: Optional callback function(model_name, success)

        Returns:
            bool: True if successful
        """
        if self.is_model_cached(model_name):
            if self.debug_mode:
                print(f"✅ {model_name} already cached")
            if progress_callback:
                progress_callback(model_name, True)
            return True

        try:
            TTS, _ = _import_tts()

            if self.debug_mode:
                print(f"📥 Downloading {model_name}...")

            start_time = time.time()

            # Initialize TTS to trigger download
            tts = TTS(model_name=model_name, progress_bar=True)

            download_time = time.time() - start_time
            if self.debug_mode:
                print(f"✅ Downloaded {model_name} in {download_time:.1f}s")

            if progress_callback:
                progress_callback(model_name, True)
            return True

        except Exception as e:
            if self.debug_mode:
                print(f"❌ Failed to download {model_name}: {e}")
            if progress_callback:
                progress_callback(model_name, False)
            return False

    def download_essential_model(self, progress_callback: Optional[Callable[[str, bool], None]] = None) -> bool:
        """Download the essential English model for immediate functionality."""
        return self.download_model(self.ESSENTIAL_MODEL, progress_callback)

    def list_available_models(self, language: Optional[str] = None) -> Dict[str, Any]:
        """Get list of available models with metadata.

        Args:
            language: Optional language filter

        Returns:
            dict: Model information in JSON-serializable format
        """
        if language:
            if language in self.AVAILABLE_MODELS:
                return {language: self.AVAILABLE_MODELS[language]}
            else:
                return {}

        # Return all models with cache status
        result = {}
        for lang, models in self.AVAILABLE_MODELS.items():
            result[lang] = {}
            for model_id, model_info in models.items():
                # Add cache status to each model
                model_data = model_info.copy()
                model_data["cached"] = self.is_model_cached(model_info["model"])
                result[lang][model_id] = model_data

        return result

    def get_cached_models(self) -> List[str]:
        """Get list of model names that are currently cached."""
        if not os.path.exists(self.cache_dir):
            return []

        cached = []
        try:
            for item in os.listdir(self.cache_dir):
                if item.startswith("tts_models--"):
                    # Convert cache name back to model name
                    model_name = item.replace("--", "/")
                    if self.is_model_cached(model_name):
                        cached.append(model_name)
        except Exception as e:
            if self.debug_mode:
                print(f"Error listing cached models: {e}")

        return cached

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status information."""
        cached_models = self.get_cached_models()
        essential_cached = self.ESSENTIAL_MODEL in cached_models

        # Calculate total cache size
        total_size_mb = 0
        if os.path.exists(self.cache_dir):
            try:
                for root, dirs, files in os.walk(self.cache_dir):
                    for file in files:
                        total_size_mb += os.path.getsize(os.path.join(root, file)) / (1024 * 1024)
            except:
                pass

        return {
            "cache_dir": self.cache_dir,
            "cached_models": cached_models,
            "total_cached": len(cached_models),
            "essential_model_cached": essential_cached,
            "essential_model": self.ESSENTIAL_MODEL,
            "ready_for_offline": essential_cached,
            "total_size_mb": round(total_size_mb, 1),
            "available_languages": list(self.AVAILABLE_MODELS.keys()),
        }

    def clear_cache(self, confirm: bool = False) -> bool:
        """Clear the model cache."""
        if not confirm:
            return False

        try:
            import shutil
            if os.path.exists(self.cache_dir):
                shutil.rmtree(self.cache_dir)
                if self.debug_mode:
                    print(f"✅ Cleared model cache: {self.cache_dir}")
                return True
            return True
        except Exception as e:
            if self.debug_mode:
                print(f"❌ Failed to clear cache: {e}")
            return False

    def ensure_essential_model(self, auto_download: bool = True) -> bool:
        """Ensure the essential model is available.

        Args:
            auto_download: Whether to download if not cached

        Returns:
            bool: True if essential model is ready
        """
        if self.is_model_cached(self.ESSENTIAL_MODEL):
            return True

        if not auto_download:
            return False

        return self.download_essential_model()


# Global instance for easy access
_model_manager = None

def get_model_manager(debug_mode: bool = False) -> SimpleModelManager:
    """Get the global model manager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = SimpleModelManager(debug_mode=debug_mode)
    return _model_manager


# Simple API functions for third-party use
def list_models(language: Optional[str] = None) -> str:
    """Get available models as JSON string.

    Args:
        language: Optional language filter

    Returns:
        str: JSON string of available models
    """
    manager = get_model_manager()
    return json.dumps(manager.list_available_models(language), indent=2)


def download_model(model_name: str, progress_callback: Optional[Callable[[str, bool], None]] = None) -> bool:
    """Download a specific model.

    Args:
        model_name: Model name or voice ID (e.g., 'en.vits' or 'tts_models/en/ljspeech/vits')
        progress_callback: Optional progress callback

    Returns:
        bool: True if successful
    """
    manager = get_model_manager()

    # Handle voice ID format (e.g., 'en.vits')
    if '.' in model_name and not model_name.startswith('tts_models'):
        lang, voice_id = model_name.split('.', 1)
        if lang in manager.AVAILABLE_MODELS and voice_id in manager.AVAILABLE_MODELS[lang]:
            model_name = manager.AVAILABLE_MODELS[lang][voice_id]["model"]
        else:
            return False

    return manager.download_model(model_name, progress_callback)


def get_status() -> str:
    """Get model cache status as JSON string."""
    manager = get_model_manager()
    return json.dumps(manager.get_status(), indent=2)


def is_ready() -> bool:
    """Check if essential model is ready for immediate use."""
    manager = get_model_manager()
    return manager.is_model_cached(manager.ESSENTIAL_MODEL)