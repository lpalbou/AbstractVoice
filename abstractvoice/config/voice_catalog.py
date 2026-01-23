"""Language metadata (Piper-first).

AbstractVoice core uses Piper as the default (and only) TTS engine. We keep a
small language list here for validation / UX messaging. Voice selection is
handled by the Piper adapter itself.
"""

LANGUAGES = {
    "en": {"name": "English"},
    "fr": {"name": "French"},
    "de": {"name": "German"},
    "es": {"name": "Spanish"},
    "ru": {"name": "Russian"},
    "zh": {"name": "Chinese"},
}

# Universal safe fallback language code.
SAFE_FALLBACK = "en"

