"""Common helpers for VoiceManager parts.

This module exists to avoid circular imports while keeping `voice_manager.py`
small and focused on the public façade.
"""

from __future__ import annotations

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

