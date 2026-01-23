"""Common helpers for VoiceManager parts.

This module exists to avoid circular imports while keeping `voice_manager.py`
small and focused on the public façade.
"""

from __future__ import annotations

from typing import Type


def import_tts_engine():
    """Import TTSEngine with a helpful error if optional deps are missing."""
    try:
        from ..tts import TTSEngine
        return TTSEngine
    except ImportError as e:
        if "TTS" in str(e) or "torch" in str(e) or "librosa" in str(e):
            raise ImportError(
                "TTS functionality requires optional dependencies. Install with:\n"
                "  pip install abstractvoice[tts]    # For TTS only\n"
                "  pip install abstractvoice[all]    # For all features\n"
                f"Original error: {e}"
            ) from e
        raise


def import_voice_recognizer():
    """Import VoiceRecognizer with a helpful error if dependencies are missing."""
    try:
        from ..recognition import VoiceRecognizer
        return VoiceRecognizer
    except ImportError as e:
        raise ImportError(
            "Microphone capture/listen() requires optional dependencies to be installed correctly.\n"
            "Try:\n"
            "  pip install --upgrade abstractvoice\n"
            f"Original error: {e}"
        ) from e

