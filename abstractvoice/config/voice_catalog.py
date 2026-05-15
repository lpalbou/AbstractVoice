"""Language metadata for local TTS engines.

Remote engines treat language as a provider hint. Piper validates against its
own small curated mapping; broader local engines such as Supertonic use this
catalog for display / UX messaging.
"""

LANGUAGES = {
    "ar": {"name": "Arabic"},
    "bg": {"name": "Bulgarian"},
    "cs": {"name": "Czech"},
    "da": {"name": "Danish"},
    "de": {"name": "German"},
    "el": {"name": "Greek"},
    "en": {"name": "English"},
    "es": {"name": "Spanish"},
    "et": {"name": "Estonian"},
    "fi": {"name": "Finnish"},
    "fr": {"name": "French"},
    "hi": {"name": "Hindi"},
    "hr": {"name": "Croatian"},
    "hu": {"name": "Hungarian"},
    "id": {"name": "Indonesian"},
    "it": {"name": "Italian"},
    "ja": {"name": "Japanese"},
    "ko": {"name": "Korean"},
    "lt": {"name": "Lithuanian"},
    "lv": {"name": "Latvian"},
    "nl": {"name": "Dutch"},
    "pl": {"name": "Polish"},
    "pt": {"name": "Portuguese"},
    "ro": {"name": "Romanian"},
    "ru": {"name": "Russian"},
    "sk": {"name": "Slovak"},
    "sl": {"name": "Slovenian"},
    "sv": {"name": "Swedish"},
    "tr": {"name": "Turkish"},
    "uk": {"name": "Ukrainian"},
    "vi": {"name": "Vietnamese"},
    "zh": {"name": "Chinese"},
}

# Universal safe fallback language code.
SAFE_FALLBACK = "en"
