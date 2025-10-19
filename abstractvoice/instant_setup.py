"""
Instant Setup Module for AbstractVoice
Provides immediate TTS functionality with seamless model download.
"""

import os
import sys
from pathlib import Path

# Essential model for instant functionality (lightweight, reliable)
ESSENTIAL_MODEL = "tts_models/en/ljspeech/fast_pitch"
ESSENTIAL_MODEL_SIZE = "~100MB"

def ensure_instant_tts():
    """
    Ensure TTS is ready for immediate use.
    Downloads essential model if needed with progress indicator.

    Returns:
        bool: True if TTS is ready, False if failed
    """
    try:
        from TTS.api import TTS
        from TTS.utils.manage import ModelManager

        manager = ModelManager()

        # Check if essential model is already cached
        if is_model_cached(ESSENTIAL_MODEL):
            return True

        # Download essential model with user-friendly progress
        print(f"🚀 AbstractVoice: Setting up TTS ({ESSENTIAL_MODEL_SIZE})...")
        print(f"   This happens once and takes ~30 seconds")

        try:
            # Download with progress bar
            tts = TTS(model_name=ESSENTIAL_MODEL, progress_bar=True)
            print(f"✅ TTS ready! AbstractVoice is now fully functional.")
            return True

        except Exception as e:
            print(f"❌ Setup failed: {e}")
            print(f"💡 Try: pip install abstractvoice[all]")
            return False

    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print(f"💡 Install with: pip install abstractvoice[all]")
        return False

def is_model_cached(model_name):
    """Check if a model is already cached."""
    try:
        from TTS.utils.manage import ModelManager
        manager = ModelManager()

        # Get cached models list
        models_file = os.path.join(manager.output_prefix, ".models.json")
        if os.path.exists(models_file):
            import json
            with open(models_file, 'r') as f:
                cached_models = json.load(f)
                return model_name in cached_models

        # Fallback: check if model directory exists and has content
        model_dir = model_name.replace("/", "--")
        model_path = os.path.join(manager.output_prefix, model_dir)
        return os.path.exists(model_path) and bool(os.listdir(model_path))

    except:
        # If anything fails, assume not cached
        return False

def get_instant_model():
    """Get the essential model name for instant setup."""
    return ESSENTIAL_MODEL

if __name__ == "__main__":
    # CLI test
    print("🧪 Testing instant setup...")
    success = ensure_instant_tts()
    print(f"Result: {'✅ Ready' if success else '❌ Failed'}")