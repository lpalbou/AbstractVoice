"""Model management utilities for AbstractVoice.

This module provides utilities for downloading, caching, and managing TTS models
to ensure offline functionality and better user experience.
"""

import os
import sys
import time
import threading
from typing import List, Optional, Dict, Any
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
            "  pip install abstractvoice[tts]        # For TTS only\n"
            "  pip install abstractvoice[voice-full] # For complete voice functionality\n"
            "  pip install abstractvoice[all]        # For all features\n"
            f"Original error: {e}"
        ) from e


class ModelManager:
    """Manages TTS model downloading, caching, and offline availability."""

    # Essential models for immediate functionality
    ESSENTIAL_MODELS = [
        "tts_models/en/ljspeech/fast_pitch",     # Lightweight, no espeak dependency
        "tts_models/en/ljspeech/tacotron2-DDC",  # Reliable fallback
    ]

    # Premium models for best quality (downloaded on-demand)
    PREMIUM_MODELS = [
        "tts_models/en/ljspeech/vits",           # Best quality English
        "tts_models/fr/css10/vits",              # Best quality French
        "tts_models/es/mai/tacotron2-DDC",       # Best quality Spanish
        "tts_models/de/thorsten/vits",           # Best quality German
        "tts_models/it/mai_male/vits",           # Best quality Italian
    ]

    # All supported models
    ALL_MODELS = ESSENTIAL_MODELS + PREMIUM_MODELS

    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
        self._cache_dir = None
        self._model_manager = None

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

    @property
    def model_manager(self):
        """Get TTS ModelManager instance."""
        if self._model_manager is None:
            _, ModelManagerClass = _import_tts()
            self._model_manager = ModelManagerClass()
        return self._model_manager

    def check_model_cache(self, model_name: str) -> bool:
        """Check if a model is already cached locally."""
        try:
            # Look for model files in cache
            model_path = self._get_model_path(model_name)
            if model_path and os.path.exists(model_path):
                # Check for essential model files
                model_files = ["model.pth", "config.json"]
                return any(
                    os.path.exists(os.path.join(model_path, f))
                    for f in model_files
                )
            return False
        except Exception as e:
            if self.debug_mode:
                print(f"Error checking cache for {model_name}: {e}")
            return False

    def _get_model_path(self, model_name: str) -> Optional[str]:
        """Get the expected cache path for a model."""
        # Convert model name to cache directory structure
        # e.g., "tts_models/en/ljspeech/vits" -> "tts_models--en--ljspeech--vits"
        cache_name = model_name.replace("/", "--")
        return os.path.join(self.cache_dir, cache_name)

    def get_cached_models(self) -> List[str]:
        """Get list of models that are cached locally."""
        if not os.path.exists(self.cache_dir):
            return []

        cached = []
        try:
            for item in os.listdir(self.cache_dir):
                if item.startswith("tts_models--"):
                    # Convert cache name back to model name
                    model_name = item.replace("--", "/")
                    if self.check_model_cache(model_name):
                        cached.append(model_name)
        except Exception as e:
            if self.debug_mode:
                print(f"Error listing cached models: {e}")

        return cached

    def download_model(self, model_name: str, force: bool = False) -> bool:
        """Download a specific model."""
        if not force and self.check_model_cache(model_name):
            if self.debug_mode:
                print(f"✅ {model_name} already cached")
            return True

        try:
            TTS, _ = _import_tts()

            print(f"📥 Downloading {model_name}...")
            start_time = time.time()

            # Initialize TTS to trigger download
            tts = TTS(model_name=model_name, progress_bar=True)

            download_time = time.time() - start_time
            print(f"✅ Downloaded {model_name} in {download_time:.1f}s")
            return True

        except Exception as e:
            print(f"❌ Failed to download {model_name}: {e}")
            return False

    def download_all_models(self) -> bool:
        """Download all supported models."""
        print("📦 Downloading all TTS models...")

        success_count = 0
        for model in self.ALL_MODELS:
            if self.download_model(model):
                success_count += 1

        print(f"✅ Downloaded {success_count}/{len(self.ALL_MODELS)} models")
        return success_count > 0

    def get_offline_model(self, preferred_models: List[str]) -> Optional[str]:
        """Get the best available cached model from a preference list."""
        cached_models = self.get_cached_models()

        # Return first preferred model that's cached
        for model in preferred_models:
            if model in cached_models:
                return model

        # Fallback to any cached model
        if cached_models:
            return cached_models[0]

        return None

    def print_status(self):
        """Print current model cache status."""
        print("🎭 TTS Model Cache Status")
        print("=" * 50)

        cached_models = self.get_cached_models()

        if not cached_models:
            print("❌ No models cached - first use will require internet")
            print("\nTo download essential models for offline use:")
            print("  abstractvoice download-models")
            return

        print(f"✅ {len(cached_models)} models cached for offline use:")

        # Group by category
        essential_cached = [m for m in cached_models if m in self.ESSENTIAL_MODELS]
        premium_cached = [m for m in cached_models if m in self.PREMIUM_MODELS]
        other_cached = [m for m in cached_models if m not in self.ALL_MODELS]

        if essential_cached:
            print(f"\n📦 Essential Models ({len(essential_cached)}):")
            for model in essential_cached:
                print(f"  ✅ {model}")

        if premium_cached:
            print(f"\n✨ Premium Models ({len(premium_cached)}):")
            for model in premium_cached:
                print(f"  ✅ {model}")

        if other_cached:
            print(f"\n🔧 Other Models ({len(other_cached)}):")
            for model in other_cached:
                print(f"  ✅ {model}")

        print(f"\n💾 Cache location: {self.cache_dir}")

        # Check cache size
        try:
            total_size = 0
            for root, dirs, files in os.walk(self.cache_dir):
                for file in files:
                    total_size += os.path.getsize(os.path.join(root, file))
            size_mb = total_size / (1024 * 1024)
            print(f"💽 Total cache size: {size_mb:.1f} MB")
        except:
            pass

    def clear_cache(self, confirm: bool = False) -> bool:
        """Clear the model cache."""
        if not confirm:
            print("⚠️ This will delete all cached TTS models.")
            print("Use clear_cache(confirm=True) to proceed.")
            return False

        try:
            import shutil
            if os.path.exists(self.cache_dir):
                shutil.rmtree(self.cache_dir)
                print(f"✅ Cleared model cache: {self.cache_dir}")
                return True
            else:
                print("ℹ️ No cache to clear")
                return True
        except Exception as e:
            print(f"❌ Failed to clear cache: {e}")
            return False


def download_models_cli():
    """CLI entry point for downloading models."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Download TTS models for offline use")
    parser.add_argument("--essential", action="store_true",
                       help="Download only essential models (recommended)")
    parser.add_argument("--all", action="store_true",
                       help="Download all supported models")
    parser.add_argument("--model", type=str,
                       help="Download specific model by name")
    parser.add_argument("--language", type=str,
                       help="Download models for specific language (en, fr, es, de, it)")
    parser.add_argument("--status", action="store_true",
                       help="Show current cache status")
    parser.add_argument("--clear", action="store_true",
                       help="Clear model cache")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug output")

    args = parser.parse_args()

    # Use VoiceManager for consistent programmatic API
    from abstractvoice.voice_manager import VoiceManager

    vm = VoiceManager(debug_mode=args.debug)

    if args.status:
        # Use VoiceManager's model status
        status = vm.get_cache_status()
        print("🎭 TTS Model Cache Status")
        print("=" * 50)

        if status['total_cached'] == 0:
            print("❌ No models cached - first use will require internet")
            print("\nTo download essential models for offline use:")
            print("  abstractvoice download-models --essential")
            return

        print(f"✅ {status['total_cached']} models cached for offline use")
        print(f"📦 Essential model cached: {status['essential_model_cached']}")
        print(f"🌐 Ready for offline: {status['ready_for_offline']}")
        print(f"💾 Cache location: {status['cache_dir']}")
        print(f"💽 Total cache size: {status['total_size_mb']} MB")

        # Show cached models
        cached_models = status['cached_models']
        essential_model = status['essential_model']

        print(f"\n📦 Essential Model:")
        if essential_model in cached_models:
            print(f"  ✅ {essential_model}")
        else:
            print(f"  📥 {essential_model} (not cached)")

        print(f"\n📋 All Cached Models ({len(cached_models)}):")
        for model in sorted(cached_models)[:10]:  # Show first 10
            print(f"  ✅ {model}")
        if len(cached_models) > 10:
            print(f"  ... and {len(cached_models) - 10} more")
        return

    if args.clear:
        # Use ModelManager for low-level cache operations
        manager = ModelManager(debug_mode=args.debug)
        manager.clear_cache(confirm=True)
        return

    if args.model:
        # Use ModelManager for direct model download
        manager = ModelManager(debug_mode=args.debug)
        success = manager.download_model(args.model)
        sys.exit(0 if success else 1)

    if args.language:
        # Use simple model download for language-specific models
        print(f"📦 Downloading models for {args.language}...")

        # Get available models for this language
        models = vm.list_available_models(args.language)
        if args.language not in models:
            print(f"❌ Language '{args.language}' not supported")
            print(f"   Available languages: {list(vm.list_available_models().keys())}")
            sys.exit(1)

        # Download the default model for this language
        language_models = models[args.language]
        default_model = None
        for voice_id, voice_info in language_models.items():
            if voice_info.get('default', False):
                default_model = f"{args.language}.{voice_id}"
                break

        if not default_model:
            # Take the first available model
            first_voice = list(language_models.keys())[0]
            default_model = f"{args.language}.{first_voice}"

        print(f"  📥 Downloading {default_model}...")
        success = vm.download_model(default_model)

        if success:
            print(f"✅ Downloaded {default_model}")
            print(f"✅ {args.language.upper()} voice is now ready!")
        else:
            print(f"❌ Failed to download {default_model}")
        sys.exit(0 if success else 1)

    if args.all:
        # Use ModelManager for downloading all models
        manager = ModelManager(debug_mode=args.debug)
        success = manager.download_all_models()
        sys.exit(0 if success else 1)

    # Default to essential models via VoiceManager
    if args.essential or (not args.all and not args.model and not args.language):
        print("📦 Downloading essential TTS model for offline use...")

        # Use the simple ensure_ready method
        success = vm.ensure_ready(auto_download=True)

        if success:
            print("✅ Essential model downloaded successfully!")
            print("🎉 AbstractVoice is now ready for offline use!")
        else:
            print("❌ Essential model download failed")
            print("   Check your internet connection")
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    download_models_cli()