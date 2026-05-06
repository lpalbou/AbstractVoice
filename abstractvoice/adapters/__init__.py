"""Adapter interfaces for TTS and STT engines.

This module defines base interfaces for pluggable TTS and STT engines,
enabling easy integration of new speech synthesis and recognition backends
while maintaining API compatibility.
"""

from .base import TTSAdapter, STTAdapter
from .tts_piper import PiperTTSAdapter
from .stt_faster_whisper import FasterWhisperAdapter
from .tts_openai_compatible import OpenAICompatibleTTSAdapter
from .stt_openai_compatible import OpenAICompatibleSTTAdapter

__all__ = [
    'TTSAdapter',
    'STTAdapter',
    'PiperTTSAdapter',
    'FasterWhisperAdapter',
    'OpenAICompatibleTTSAdapter',
    'OpenAICompatibleSTTAdapter',
]
