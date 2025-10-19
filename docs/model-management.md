# Model Management Guide

This guide covers everything you need to know about managing TTS models in AbstractVoice v0.4.0+.

## 🎯 Quick Start

AbstractVoice automatically downloads essential models on first use. For most users, no additional setup is needed:

```python
from abstractvoice import VoiceManager

vm = VoiceManager()
vm.speak("Hello! TTS works immediately!")  # Downloads essential model if needed
```

## 📦 Model Types

### Essential Model
- **Model**: `tts_models/en/ljspeech/fast_pitch` (107MB)
- **Purpose**: Guaranteed to work everywhere, lightweight, reliable
- **Auto-download**: Yes, on first TTS use
- **Quality**: Good English voice

### Language Models
- **French**: CSS10 VITS (548MB) - High quality
- **Spanish**: MAI Tacotron2 (362MB) - Reliable
- **German**: Thorsten VITS (548MB) - High quality
- **Italian**: MAI Male VITS (548MB) - High quality
- **Auto-download**: No, on-demand only

## 🔧 Programmatic APIs

### Simple JSON APIs (for third-party applications)

```python
from abstractvoice import list_models, download_model, get_status, is_ready

# Check if system is ready
ready = is_ready()  # Returns: True/False

# Get all models as JSON
models_json = list_models()  # All languages
models_json = list_models('fr')  # Just French

# Get cache status
status_json = get_status()
# Returns: {"total_cached": 17, "ready_for_offline": true, ...}

# Download specific model
success = download_model('fr.css10_vits')  # Voice ID format
success = download_model('tts_models/fr/css10/vits')  # Full name
```

### VoiceManager APIs (for library integration)

```python
from abstractvoice import VoiceManager

vm = VoiceManager()

# Check if ready for immediate use
ready = vm.is_model_ready()

# Ensure TTS is ready (downloads if needed)
ready = vm.ensure_ready(auto_download=True)

# List available models with metadata
models = vm.list_available_models()  # All languages
models = vm.list_available_models('fr')  # Just French

# Download specific model
success = vm.download_model('de.thorsten_vits')

# Get cache status
status = vm.get_cache_status()
```

## 💻 CLI Commands

### Download Models

```bash
# Download essential model (recommended first step)
abstractvoice download-models

# Download all models for a language
abstractvoice download-models --language fr
abstractvoice download-models --language de

# Download specific model
abstractvoice download-models --model tts_models/fr/css10/vits

# Download all available models (large!)
abstractvoice download-models --all

# Check current status
abstractvoice download-models --status

# Clear cache
abstractvoice download-models --clear
```

### Voice Selection in CLI

```bash
# Start CLI
abstractvoice cli

# In CLI - list all available voices
/setvoice

# In CLI - set specific voice (downloads if needed)
/setvoice fr.css10_vits
/setvoice de.thorsten_vits
/setvoice it.mai_female_vits
```

## 🌐 Model Information

### English Models

| Voice ID | Model Name | Size | Quality | Dependencies |
|----------|------------|------|---------|--------------|
| `en.fast_pitch` | Fast Pitch (English) | 107MB | Good | None |
| `en.vits` | VITS (English) | 328MB | Excellent | espeak-ng |
| `en.tacotron2` | Tacotron2 (English) | 362MB | Good | None |

### French Models

| Voice ID | Model Name | Size | Quality | Dependencies |
|----------|------------|------|---------|--------------|
| `fr.css10_vits` | CSS10 VITS (French) | 548MB | Excellent | espeak-ng |
| `fr.mai_tacotron2` | MAI Tacotron2 (French) | 362MB | Good | None |

### Spanish Models

| Voice ID | Model Name | Size | Quality | Dependencies |
|----------|------------|------|---------|--------------|
| `es.mai_tacotron2` | MAI Tacotron2 (Spanish) | 362MB | Good | None |
| `es.css10_vits` | CSS10 VITS (Spanish) | 548MB | Excellent | espeak-ng |

### German Models

| Voice ID | Model Name | Size | Quality | Dependencies |
|----------|------------|------|---------|--------------|
| `de.thorsten_vits` | Thorsten VITS (German) | 548MB | Excellent | espeak-ng |

### Italian Models

| Voice ID | Model Name | Size | Quality | Dependencies |
|----------|------------|------|---------|--------------|
| `it.mai_male_vits` | MAI Male VITS (Italian) | 548MB | Excellent | espeak-ng |
| `it.mai_female_vits` | MAI Female VITS (Italian) | 548MB | Excellent | espeak-ng |

## 📂 Cache Management

### Cache Locations

Models are cached in platform-specific locations:

- **macOS**: `~/Library/Application Support/tts`
- **Linux**: `~/.local/share/tts` or `~/.cache/tts`
- **Windows**: `%APPDATA%\tts`

### Cache Status

```python
# Get detailed cache information
from abstractvoice import get_status
import json

status = json.loads(get_status())
print(f"Total models: {status['total_cached']}")
print(f"Cache size: {status['total_size_mb']} MB")
print(f"Ready for offline: {status['ready_for_offline']}")
```

## 🔄 Integration Patterns

### Simple Integration

```python
from abstractvoice import VoiceManager

# One-liner: ensure models are ready and use TTS
vm = VoiceManager()
if vm.ensure_ready():
    vm.speak("Ready to go!")
```

### Robust Integration

```python
from abstractvoice import VoiceManager

def setup_voice_system():
    vm = VoiceManager()

    # Check if ready
    if vm.is_model_ready():
        print("✅ TTS ready immediately")
        return vm

    # Download essential model
    print("📥 Downloading essential model...")
    if vm.ensure_ready():
        print("✅ TTS ready!")
        return vm
    else:
        print("❌ TTS setup failed")
        return None

vm = setup_voice_system()
if vm:
    vm.speak("System initialized successfully!")
```

### Enterprise Integration

```python
from abstractvoice import VoiceManager

def verify_production_readiness():
    """Verify all required models are available before deployment."""
    vm = VoiceManager()
    required_languages = ['en', 'fr', 'es', 'de']

    print("🔍 Verifying production readiness...")
    all_ready = True

    for lang in required_languages:
        models = vm.list_available_models(lang)
        cached_count = sum(1 for voice in models[lang].values() if voice.get('cached', False))

        print(f"  {lang.upper()}: {cached_count} models ready")
        if cached_count == 0:
            all_ready = False

    return all_ready

# In deployment script
if not verify_production_readiness():
    print("❌ Missing required models for production")
    exit(1)
```

## 🌐 Web API Integration

```python
# Example Flask API endpoints
from flask import Flask, jsonify
from abstractvoice import list_models, download_model, get_status, is_ready

app = Flask(__name__)

@app.route('/api/tts/status')
def tts_status():
    return jsonify({"ready": is_ready()})

@app.route('/api/tts/models')
def list_tts_models():
    return list_models()  # Returns JSON string

@app.route('/api/tts/models/<language>')
def list_language_models(language):
    return list_models(language)

@app.route('/api/tts/download/<voice_id>', methods=['POST'])
def download_voice(voice_id):
    success = download_model(voice_id)
    return jsonify({"success": success})

@app.route('/api/tts/cache/status')
def cache_status():
    return get_status()  # Returns JSON string
```

## 🚀 Performance Tips

### Fast Initialization

1. **Check readiness first**: Use `is_ready()` before creating VoiceManager
2. **Cache VoiceManager**: Create once, reuse across requests
3. **Pre-download models**: Use CLI commands in deployment scripts

### Model Selection

1. **Start with essential**: Fast Pitch works everywhere
2. **Upgrade selectively**: Download VITS only for high-quality needs
3. **Language-specific**: Download only languages you need

### Production Deployment

```bash
# In your deployment script
abstractvoice download-models --essential
abstractvoice download-models --language fr
abstractvoice download-models --language es

# Verify readiness
abstractvoice download-models --status
```

## ❓ Troubleshooting

### Model Download Fails

```bash
# Check internet connectivity
curl -I https://coqui.gateway.scarf.sh

# Clear corrupted cache
abstractvoice download-models --clear

# Try essential model only
abstractvoice download-models --essential
```

### Cache Issues

```python
# Check cache status
from abstractvoice import get_status
import json

status = json.loads(get_status())
print(f"Cache dir: {status['cache_dir']}")
print(f"Models: {len(status['cached_models'])}")
```

### espeak-ng Issues

For VITS models requiring espeak-ng:

```bash
# macOS
brew install espeak-ng

# Ubuntu/Debian
sudo apt-get install espeak-ng

# Alternative: Use non-VITS models
abstractvoice download-models --model tts_models/en/ljspeech/fast_pitch
```

## 📚 Examples

See the complete examples in:
- [`examples/library_integration.py`](../examples/library_integration.py) - Library integration patterns
- [`examples/third_party_api_demo.py`](../examples/third_party_api_demo.py) - Third-party API usage