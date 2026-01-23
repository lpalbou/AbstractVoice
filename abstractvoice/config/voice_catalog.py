"""Language and voice catalog constants.

This module centralizes “product” configuration (supported languages, defaults,
voice metadata) so `VoiceManager` can stay focused on orchestration logic.
"""

LANGUAGES = {
    "en": {
        "default": "tts_models/en/ljspeech/tacotron2-DDC",
        "premium": "tts_models/en/ljspeech/vits",
        "name": "English",
    },
    "fr": {
        "default": "tts_models/fr/css10/vits",
        "premium": "tts_models/fr/css10/vits",
        "name": "French",
    },
    "es": {
        "default": "tts_models/es/mai/tacotron2-DDC",
        "premium": "tts_models/es/mai/tacotron2-DDC",
        "name": "Spanish",
    },
    "de": {
        "default": "tts_models/de/thorsten/vits",
        "premium": "tts_models/de/thorsten/vits",
        "name": "German",
    },
    "it": {
        "default": "tts_models/it/mai_male/vits",
        "premium": "tts_models/it/mai_male/vits",
        "name": "Italian",
    },
}

# Universal safe fallback
SAFE_FALLBACK = "tts_models/en/ljspeech/fast_pitch"

# Complete voice catalog with metadata
VOICE_CATALOG = {
    "en": {
        "tacotron2": {
            "model": "tts_models/en/ljspeech/tacotron2-DDC",
            "quality": "good",
            "gender": "female",
            "accent": "US English",
            "license": "Open source (LJSpeech)",
            "requires": "none",
        },
        "jenny": {
            "model": "tts_models/en/jenny/jenny",
            "quality": "excellent",
            "gender": "female",
            "accent": "US English",
            "license": "Open source (Jenny)",
            "requires": "none",
        },
        "ek1": {
            "model": "tts_models/en/ek1/tacotron2",
            "quality": "excellent",
            "gender": "male",
            "accent": "British English",
            "license": "Open source (EK1)",
            "requires": "none",
        },
        "sam": {
            "model": "tts_models/en/sam/tacotron-DDC",
            "quality": "good",
            "gender": "male",
            "accent": "US English",
            "license": "Open source (Sam)",
            "requires": "none",
        },
        "fast_pitch": {
            "model": "tts_models/en/ljspeech/fast_pitch",
            "quality": "good",
            "gender": "female",
            "accent": "US English",
            "license": "Open source (LJSpeech)",
            "requires": "none",
        },
        "vits": {
            "model": "tts_models/en/ljspeech/vits",
            "quality": "premium",
            "gender": "female",
            "accent": "US English",
            "license": "Open source (LJSpeech)",
            "requires": "espeak-ng",
        },
    },
    "fr": {
        "css10_vits": {
            "model": "tts_models/fr/css10/vits",
            "quality": "premium",
            "gender": "male",
            "accent": "France French",
            "license": "Apache 2.0 (CSS10/LibriVox)",
            "requires": "espeak-ng",
        },
        "mai_tacotron": {
            "model": "tts_models/fr/mai/tacotron2-DDC",
            "quality": "good",
            "gender": "female",
            "accent": "France French",
            "license": "Permissive (M-AILABS/LibriVox)",
            "requires": "none",
        },
    },
    "es": {
        "mai_tacotron": {
            "model": "tts_models/es/mai/tacotron2-DDC",
            "quality": "good",
            "gender": "female",
            "accent": "Spain Spanish",
            "license": "Permissive (M-AILABS)",
            "requires": "none",
        }
    },
    "de": {
        "thorsten_vits": {
            "model": "tts_models/de/thorsten/vits",
            "quality": "premium",
            "gender": "male",
            "accent": "Standard German",
            "license": "Open source (Thorsten)",
            "requires": "espeak-ng",
        },
        "thorsten_tacotron": {
            "model": "tts_models/de/thorsten/tacotron2-DDC",
            "quality": "good",
            "gender": "male",
            "accent": "Standard German",
            "license": "Open source (Thorsten)",
            "requires": "none",
        },
    },
    "it": {
        "mai_male_vits": {
            "model": "tts_models/it/mai_male/vits",
            "quality": "premium",
            "gender": "male",
            "accent": "Standard Italian",
            "license": "Permissive (M-AILABS)",
            "requires": "espeak-ng",
            "speed": 0.8,  # Slow down to fix pace issues
        },
        "mai_female_vits": {
            "model": "tts_models/it/mai_female/vits",
            "quality": "premium",
            "gender": "female",
            "accent": "Standard Italian",
            "license": "Permissive (M-AILABS)",
            "requires": "espeak-ng",
            "speed": 0.8,  # Slow down to fix pace issues
        },
    },
}

