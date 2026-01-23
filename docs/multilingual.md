# Multilingual support (current)

## TTS (Piper, default)

Piper voices are currently configured per-language. Supported language codes in the default mapping include:

- `en`, `fr`, `de`, `es`, `ru`, `zh`

Usage:

```python
from abstractvoice import VoiceManager

vm = VoiceManager(language="en")
vm.speak("Hello")

vm.set_language("fr")
vm.speak("Bonjour")
```

## STT (faster-whisper, default)

STT supports many languages. You can pass a language hint when transcribing files:

```python
text = vm.transcribe_file("audio.wav", language="fr")
```

## Notes

- Available languages/voices can evolve over time; see `abstractvoice/adapters/tts_piper.py` for the current mapping.
- For commercial use, verify voice/model licensing (see `docs/voices-and-licenses.md`).

### Simple Language Selection

```python
from abstractvoice import VoiceManager

# Create voice manager for specific language
vm_fr = VoiceManager(language='fr')
vm_fr.speak("Bonjour! Je suis votre assistant IA.")

vm_es = VoiceManager(language='es')
vm_es.speak("¡Hola! Soy tu asistente de IA.")

vm_de = VoiceManager(language='de')
vm_de.speak("Hallo! Ich bin Ihr KI-Assistent.")

vm_it = VoiceManager(language='it')
vm_it.speak("Ciao! Sono il tuo assistente IA.")

vm_ru = VoiceManager(language='ru')
vm_ru.speak("Привет! Я ваш ИИ-помощник.")
```

### Language Convenience Functions

```python
from abstractvoice import (
    VoiceManagerFrench,
    VoiceManagerSpanish,
    VoiceManagerGerman,
    VoiceManagerItalian,
    VoiceManagerRussian
)

# Create language-specific managers
vm_fr = VoiceManagerFrench()
vm_es = VoiceManagerSpanish()
vm_de = VoiceManagerGerman()
vm_it = VoiceManagerItalian()
vm_ru = VoiceManagerRussian()

# Use them
vm_fr.speak("Comment allez-vous?")
vm_es.speak("¿Cómo estás?")
vm_de.speak("Wie geht es Ihnen?")
vm_it.speak("Come stai?")
vm_ru.speak("Как дела?")
```

### Dynamic Language Switching

```python
from abstractvoice import VoiceManagerMultilingual

# Create multilingual manager
vm = VoiceManagerMultilingual()

# Speak in different languages
vm.speak("Hello, welcome!", language='en')
vm.speak("Bonjour, bienvenue!", language='fr')
vm.speak("¡Hola, bienvenido!", language='es')
vm.speak("Hallo, willkommen!", language='de')
vm.speak("Ciao, benvenuto!", language='it')
vm.speak("Привет, добро пожаловать!", language='ru')

# Change default language
vm.set_language('fr')
vm.speak("Maintenant je parle français par défaut.")

# Get language information
info = vm.get_language_info()
print(f"Current: {info['name']} ({info['code']})")
print(f"Model: {info['model']}")
```

### Voice Conversations in Multiple Languages

```python
from abstractvoice import VoiceManager

class MultilingualAssistant:
    def __init__(self):
        self.voices = {
            'en': VoiceManager(language='en'),
            'fr': VoiceManager(language='fr'),
            'es': VoiceManager(language='es'),
            'de': VoiceManager(language='de'),
            'it': VoiceManager(language='it'),
            'ru': VoiceManager(language='ru')
        }
        self.current_language = 'en'

    def set_language(self, language):
        if language in self.voices:
            self.current_language = language
            print(f"Switched to {language}")

    def speak(self, text):
        vm = self.voices[self.current_language]
        vm.speak(text)

    def demonstrate_languages(self):
        messages = {
            'en': "Hello! I can speak in multiple languages.",
            'fr': "Bonjour! Je peux parler en plusieurs langues.",
            'es': "¡Hola! Puedo hablar en varios idiomas.",
            'de': "Hallo! Ich kann in mehreren Sprachen sprechen.",
            'it': "Ciao! Posso parlare in più lingue.",
            'ru': "Привет! Я могу говорить на нескольких языках."
        }

        for lang, message in messages.items():
            self.set_language(lang)
            self.speak(message)
            time.sleep(2)  # Pause between languages

# Usage
assistant = MultilingualAssistant()
assistant.demonstrate_languages()
```

## Advanced Features

### Speed Control with Language Support

```python
vm = VoiceManager(language='fr')

# Different speeds
vm.speak("Vitesse normale.", speed=1.0)
vm.speak("Vitesse rapide.", speed=1.5)
vm.speak("Vitesse lente.", speed=0.7)
```

### Pause/Resume in Any Language

```python
vm = VoiceManager(language='de')

# Start long speech
vm.speak("Dies ist eine sehr lange deutsche Rede, die demonstriert...")

# Pause after a moment
time.sleep(3)
vm.pause_speaking()
print("Paused German speech")

# Resume
time.sleep(2)
vm.resume_speaking()
print("Resumed German speech")
```

### Language Detection and Auto-Switching

```python
def detect_language(text):
    """Simple language detection (you can use proper libraries like langdetect)"""
    if any(word in text.lower() for word in ['bonjour', 'merci', 'français']):
        return 'fr'
    elif any(word in text.lower() for word in ['hola', 'gracias', 'español']):
        return 'es'
    elif any(word in text.lower() for word in ['hallo', 'danke', 'deutsch']):
        return 'de'
    elif any(word in text.lower() for word in ['ciao', 'grazie', 'italiano']):
        return 'it'
    elif any(word in text.lower() for word in ['привет', 'спасибо', 'русский']):
        return 'ru'
    else:
        return 'en'

def smart_speak(text):
    """Automatically detect language and speak accordingly"""
    language = detect_language(text)
    vm = VoiceManager(language=language)
    vm.speak(text)
    print(f"Detected and spoke in: {language}")

# Examples
smart_speak("Hello, how are you?")
smart_speak("Bonjour, comment allez-vous?")
smart_speak("¡Hola, cómo estás?")
smart_speak("Hallo, wie geht es dir?")
smart_speak("Ciao, come stai?")
smart_speak("Привет, как дела?")
```

## Model Information

### XTTS-v2 Model Features

The XTTS-v2 model used for multilingual support provides:

- **High Quality**: Natural-sounding speech with proper pronunciation
- **Cross-lingual**: Trained on diverse multilingual datasets
- **Consistent Voice**: Similar voice characteristics across languages
- **Real-time**: Fast synthesis suitable for interactive applications

### Model Size and Performance

```python
vm = VoiceManager(language='fr')
info = vm.get_language_info()

print(f"Language: {info['name']}")
print(f"Model: {info['model']}")
print(f"Available models: {list(info['available_models'].keys())}")

# Check if current model supports multiple languages
if 'xtts' in info['model']:
    print("✅ This model supports multiple languages")
    supported = vm.get_supported_languages()
    print(f"Supported languages: {supported}")
else:
    print("ℹ️ This is a monolingual model")
```

## Troubleshooting

### Common Issues

1. **Audio Issues on Linux**
   ```bash
   # Install audio system dependencies
   sudo apt-get install portaudio19-dev alsa-utils

   # Test audio system
   aplay /usr/share/sounds/alsa/Front_Left.wav
   ```

2. **Model Download Issues**
   ```python
   # Piper voices download on demand when you switch language.
   from abstractvoice import VoiceManager
   vm = VoiceManager()
   vm.set_language("fr")
   print("French voice ready")
   ```

3. **Language Not Working**
   ```python
   vm = VoiceManager(language='fr', debug_mode=True)
   vm.speak("Test français")  # Check debug output
   ```

### Performance Optimization

```python
# For production use with multiple languages
class OptimizedMultilingualVoice:
    def __init__(self):
        # Pre-load XTTS model once
        self.vm = VoiceManager(language='multilingual')

    def speak_in_language(self, text, language):
        # No model reloading - just change language parameter
        self.vm.speak(text, language=language)

# Usage
voice = OptimizedMultilingualVoice()
voice.speak_in_language("Hello", 'en')
voice.speak_in_language("Bonjour", 'fr')  # No reload needed
```

## Integration Examples

### Web API with Multilingual Support

```python
from flask import Flask, request, jsonify
from abstractvoice import VoiceManager

app = Flask(__name__)

# Pre-load voice managers for better performance
voice_managers = {
    'en': VoiceManager(language='en'),
    'fr': VoiceManager(language='fr'),
    'es': VoiceManager(language='es'),
    'de': VoiceManager(language='de'),
    'it': VoiceManager(language='it'),
    'ru': VoiceManager(language='ru')
}

@app.route('/speak', methods=['POST'])
def speak():
    data = request.json
    text = data.get('text', '')
    language = data.get('language', 'en')
    speed = data.get('speed', 1.0)

    if language not in voice_managers:
        return jsonify({'error': f'Language {language} not supported'}), 400

    try:
        vm = voice_managers[language]
        success = vm.speak(text, speed=speed)
        return jsonify({
            'success': success,
            'language': language,
            'text': text,
            'speed': speed
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/languages', methods=['GET'])
def get_languages():
    return jsonify({
        'supported_languages': list(voice_managers.keys()),
        'language_names': {
            'en': 'English',
            'fr': 'French',
            'es': 'Spanish',
            'de': 'German',
            'it': 'Italian',
            'ru': 'Russian'
        }
    })

if __name__ == '__main__':
    app.run(debug=True)
```

This multilingual support makes AbstractVoice a powerful choice for international applications, AI assistants, and any system requiring high-quality text-to-speech in multiple languages.