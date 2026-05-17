"""Adapter interfaces for TTS and STT engines.

This module defines base interfaces for pluggable TTS and STT engines,
enabling easy integration of new speech synthesis and recognition backends
while maintaining API compatibility.
"""

from .base import TTSAdapter, STTAdapter

__all__ = [
    'TTSAdapter',
    'STTAdapter',
    'PiperTTSAdapter',
    'SupertonicTTSAdapter',
    'FasterWhisperAdapter',
    'TransformersASRAdapter',
    'OpenAICompatibleTTSAdapter',
    'OpenAICompatibleSTTAdapter',
]


def __getattr__(name: str):
    if name == "PiperTTSAdapter":
        from .tts_piper import PiperTTSAdapter

        return PiperTTSAdapter
    if name == "SupertonicTTSAdapter":
        from .tts_supertonic import SupertonicTTSAdapter

        return SupertonicTTSAdapter
    if name == "FasterWhisperAdapter":
        from .stt_faster_whisper import FasterWhisperAdapter

        return FasterWhisperAdapter
    if name == "TransformersASRAdapter":
        from .stt_transformers_asr import TransformersASRAdapter

        return TransformersASRAdapter
    if name == "OpenAICompatibleTTSAdapter":
        from .tts_openai_compatible import OpenAICompatibleTTSAdapter

        return OpenAICompatibleTTSAdapter
    if name == "OpenAICompatibleSTTAdapter":
        from .stt_openai_compatible import OpenAICompatibleSTTAdapter

        return OpenAICompatibleSTTAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
