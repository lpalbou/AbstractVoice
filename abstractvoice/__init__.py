"""
AbstractVoice: A modular Python library for voice interactions with AI systems.

This package provides text-to-speech (TTS) and speech-to-text (STT)
capabilities with interrupt handling for AI interactions.
"""

import warnings

# Suppress the PyTorch FutureWarning about torch.load
warnings.filterwarnings(
    "ignore", 
    message="You are using `torch.load` with `weights_only=False`", 
    category=FutureWarning
)

# Suppress pkg_resources deprecation warning from jieba
warnings.filterwarnings(
    "ignore",
    message=".*pkg_resources.*",
    category=UserWarning
)
warnings.filterwarnings(
    "ignore",
    message=".*pkg_resources.*",
    category=DeprecationWarning
)

# Import the main class for public API
from .voice_manager import VoiceManager

# Import simple APIs for third-party applications
from .simple_model_manager import list_models, download_model, get_status, is_ready

__version__ = "0.5.2"
__all__ = ['VoiceManager', 'list_models', 'download_model', 'get_status', 'is_ready'] 