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
    # Changed from fast_pitch to tacotron2-DDC because fast_pitch downloads are failing
    ESSENTIAL_MODEL = "tts_models/en/ljspeech/tacotron2-DDC"

    # Available models organized by language with metadata
    AVAILABLE_MODELS = {
        "en": {
            "tacotron2": {
                "model": "tts_models/en/ljspeech/tacotron2-DDC",
                "name": "Linda (LJSpeech)",
                "quality": "good",
                "size_mb": 362,
                "description": "Standard female voice (LJSpeech speaker)",
                "requires_espeak": False,
                "default": True
            },
            "jenny": {
                "model": "tts_models/en/jenny/jenny",
                "name": "Jenny",
                "quality": "excellent",
                "size_mb": 368,
                "description": "Different female voice, clear and natural",
                "requires_espeak": False,
                "default": False
            },
            "ek1": {
                "model": "tts_models/en/ek1/tacotron2",
                "name": "Edward (EK1)",
                "quality": "excellent",
                "size_mb": 310,
                "description": "Male voice with British accent",
                "requires_espeak": False,
                "default": False
            },
            "sam": {
                "model": "tts_models/en/sam/tacotron-DDC",
                "name": "Sam",
                "quality": "good",
                "size_mb": 370,
                "description": "Different male voice, deeper tone",
                "requires_espeak": False,
                "default": False
            },
            "fast_pitch": {
                "model": "tts_models/en/ljspeech/fast_pitch",
                "name": "Linda Fast (LJSpeech)",
                "quality": "good",
                "size_mb": 107,
                "description": "Same speaker as Linda but faster engine",
                "requires_espeak": False,
                "default": False
            },
            "vits": {
                "model": "tts_models/en/ljspeech/vits",
                "name": "Linda Premium (LJSpeech)",
                "quality": "excellent",
                "size_mb": 328,
                "description": "Same speaker as Linda but premium quality",
                "requires_espeak": True,
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
        """Download a specific model with improved error handling.

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

            print(f"📥 Downloading {model_name}...")
            print(f"   This may take a few minutes depending on your connection...")

            start_time = time.time()

            # Initialize TTS to trigger download
            # Set gpu=False to avoid CUDA errors on systems without GPU
            try:
                tts = TTS(model_name=model_name, progress_bar=True, gpu=False)

                # Verify the model actually downloaded
                if not self.is_model_cached(model_name):
                    print(f"⚠️ Model download completed but not found in cache")
                    return False

            except Exception as init_error:
                # Try alternative download method
                error_msg = str(init_error).lower()
                if "connection" in error_msg or "timeout" in error_msg:
                    print(f"❌ Network error: Check your internet connection")
                elif "not found" in error_msg:
                    print(f"❌ Model '{model_name}' not found in registry")
                else:
                    print(f"❌ Download error: {init_error}")
                raise

            download_time = time.time() - start_time
            print(f"✅ Downloaded {model_name} in {download_time:.1f}s")

            if progress_callback:
                progress_callback(model_name, True)
            return True

        except Exception as e:
            error_msg = str(e).lower()

            # Provide helpful error messages
            if "connection" in error_msg or "timeout" in error_msg:
                print(f"❌ Failed to download {model_name}: Network issue")
                print(f"   Check your internet connection and try again")
            elif "permission" in error_msg:
                print(f"❌ Failed to download {model_name}: Permission denied")
                print(f"   Check write permissions for cache directory")
            elif "space" in error_msg:
                print(f"❌ Failed to download {model_name}: Insufficient disk space")
            else:
                print(f"❌ Failed to download {model_name}")
                if self.debug_mode:
                    print(f"   Error: {e}")

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


def download_models_cli():
    """Simple CLI entry point for downloading models."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Download TTS models for offline use")
    parser.add_argument("--essential", action="store_true",
                       help="Download essential model (default)")
    parser.add_argument("--all", action="store_true",
                       help="Download all available models")
    parser.add_argument("--model", type=str,
                       help="Download specific model by name")
    parser.add_argument("--language", type=str,
                       help="Download models for specific language (en, fr, es, de, it)")
    parser.add_argument("--status", action="store_true",
                       help="Show current cache status")
    parser.add_argument("--clear", action="store_true",
                       help="Clear model cache")

    args = parser.parse_args()

    manager = get_model_manager(debug_mode=True)

    if args.status:
        print(get_status())
        return

    if args.clear:
        # Ask for confirmation
        response = input("⚠️ This will delete all downloaded TTS models. Continue? (y/N): ")
        if response.lower() == 'y':
            success = manager.clear_cache(confirm=True)
            if success:
                print("✅ Model cache cleared")
            else:
                print("❌ Failed to clear cache")
        else:
            print("Cancelled")
        return

    if args.model:
        success = download_model(args.model)
        if success:
            print(f"✅ Downloaded {args.model}")
        else:
            print(f"❌ Failed to download {args.model}")
        sys.exit(0 if success else 1)

    if args.language:
        # Language-specific downloads using our simple API
        lang_models = {
            'en': ['en.tacotron2', 'en.jenny', 'en.ek1'],
            'fr': ['fr.css10_vits', 'fr.mai_tacotron2'],
            'es': ['es.mai_tacotron2'],
            'de': ['de.thorsten_vits'],
            'it': ['it.mai_male_vits', 'it.mai_female_vits']
        }

        if args.language not in lang_models:
            print(f"❌ Language '{args.language}' not supported")
            print(f"   Available: {list(lang_models.keys())}")
            sys.exit(1)

        success = False
        for model_id in lang_models[args.language]:
            if download_model(model_id):
                print(f"✅ Downloaded {model_id}")
                success = True
                break

        sys.exit(0 if success else 1)

    # Default: download essential model
    print("📦 Downloading essential TTS model...")
    success = download_model(manager.ESSENTIAL_MODEL)
    if success:
        print("✅ Essential model ready!")
    else:
        print("❌ Failed to download essential model")
    sys.exit(0 if success else 1)