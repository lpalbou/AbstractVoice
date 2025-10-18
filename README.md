# AbstractVoice

[![PyPI version](https://img.shields.io/pypi/v/abstractvoice.svg)](https://pypi.org/project/abstractvoice/)
[![Python Version](https://img.shields.io/pypi/pyversions/abstractvoice)](https://pypi.org/project/abstractvoice/)
[![License](https://img.shields.io/pypi/l/abstractvoice)](https://github.com/lpalbou/abstractvoice/blob/main/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/lpalbou/abstractvoice?style=social)](https://github.com/lpalbou/abstractvoice/stargazers)

A modular Python library for voice interactions with AI systems, providing text-to-speech (TTS) and speech-to-text (STT) capabilities with interrupt handling.

While we provide CLI and WEB examples, AbstractVoice is designed to be integrated in other projects.

## Features

- **High-Quality TTS**: Best-in-class speech synthesis with VITS model
  - Natural prosody and intonation
  - Adjustable speed without pitch distortion (using librosa time-stretching)
  - Multiple quality levels (VITS best, fast_pitch fallback)
  - Automatic fallback if espeak-ng not installed
- **Cross-Platform**: Works on macOS, Linux, and Windows
  - Best quality: Install espeak-ng (easy on all platforms)
  - Fallback mode: Works without any system dependencies
- **Speech-to-Text**: Accurate voice recognition using OpenAI's Whisper
- **Voice Activity Detection**: Efficient speech detection using WebRTC VAD
- **Interrupt Handling**: Stop TTS by speaking or using stop commands
- **Modular Design**: Easily integrate with any text generation system

## Installation

### Prerequisites

- Python 3.8+ (3.11+ recommended)
- PortAudio for audio input/output
- **Recommended**: espeak-ng for best voice quality (VITS model)

### Installing espeak-ng (Recommended for Best Quality)

AbstractVoice will work without espeak-ng, but voice quality will be significantly better with it:

**macOS:**
```bash
brew install espeak-ng
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install espeak-ng
```

**Linux (Fedora/RHEL):**
```bash
sudo yum install espeak-ng
```

**Windows:**
```bash
# Option 1: Using Conda
conda install -c conda-forge espeak-ng

# Option 2: Using Chocolatey
choco install espeak-ng

# Option 3: Download installer from https://github.com/espeak-ng/espeak-ng/releases
```

**Without espeak-ng:** AbstractVoice will automatically fall back to a simpler TTS model (fast_pitch) that works everywhere but has lower voice quality.

### Basic Installation

```bash
# Install from PyPI
pip install abstractvoice

# Or clone the repository
git clone https://github.com/lpalbou/abstractvoice.git
cd abstractvoice
pip install -e .
```

### Development Installation

```bash
# Install with development dependencies
pip install "abstractvoice[dev]"
```

### From Requirements File

```bash
# Install all dependencies including the package
pip install -r requirements.txt
```

## Quick Start

### Using AbstractVoice from the Command Line

The easiest way to get started is to use AbstractVoice directly from your shell:

```bash
# Start AbstractVoice in voice mode (TTS ON, STT ON)
abstractvoice
# → Automatically uses VITS if espeak-ng installed (best quality)
# → Falls back to fast_pitch if espeak-ng not found

# Or start with custom settings
abstractvoice --model gemma3:latest --whisper base

# Start in text-only mode (TTS enabled, listening disabled)
abstractvoice --no-listening
```

Once started, you can interact with the AI using voice or text. Use `/help` to see all available commands.

**Note**: AbstractVoice automatically selects the best available TTS model. For best quality, install espeak-ng (see Installation section above).

### Integrating AbstractVoice in Your Python Project

Here's a simple example of how to integrate AbstractVoice into your own application:

```python
from abstractvoice import VoiceManager
import time

# Initialize voice manager
voice_manager = VoiceManager(debug_mode=False)

# Text to speech
voice_manager.speak("Hello, I am an AI assistant. How can I help you today?")

# Wait for speech to complete
while voice_manager.is_speaking():
    time.sleep(0.1)

# Speech to text with callback
def on_transcription(text):
    print(f"User said: {text}")
    if text.lower() != "stop":
        # Process with your text generation system
        response = f"You said: {text}"
        voice_manager.speak(response)

# Start voice recognition
voice_manager.listen(on_transcription)

# Wait for user to say "stop" or press Ctrl+C
try:
    while voice_manager.is_listening():
        time.sleep(0.1)
except KeyboardInterrupt:
    pass

# Clean up
voice_manager.cleanup()
```

## Running Examples

The package includes several examples that demonstrate different ways to use AbstractVoice.

### Voice Mode (Default)

If installed globally, you can launch AbstractVoice directly in voice mode:

```bash
# Start AbstractVoice in voice mode (TTS ON, STT ON)
abstractvoice

# With options
abstractvoice --debug --whisper base --model gemma3:latest --api http://localhost:11434/api/chat
```

**Command line options:**
- `--debug` - Enable debug mode with detailed logging
- `--api <url>` - URL of the Ollama API (default: http://localhost:11434/api/chat)
- `--model <name>` - Ollama model to use (default: granite3.3:2b)
  - Examples: cogito:3b, phi4-mini:latest, qwen2.5:latest, gemma3:latest, etc.
- `--whisper <model>` - Whisper model to use (default: tiny)
  - Options: tiny, base, small, medium, large
- `--no-listening` - Disable speech-to-text (listening), TTS still works
  - **Note**: This creates a "TTS-only" mode where you type and the AI speaks back
- `--system <prompt>` - Custom system prompt

### Command-Line REPL

```bash
# Run the CLI example (TTS ON, STT OFF)
abstractvoice-cli cli

# With debug mode
abstractvoice-cli cli --debug
```

#### REPL Commands

All commands must start with `/` except `stop`:

**Basic Commands:**
- `/exit`, `/q`, `/quit` - Exit REPL
- `/clear` - Clear conversation history
- `/help` - Show help information
- `stop` - Stop voice mode or TTS (voice command, no `/` needed)

**Voice & Audio:**
- `/tts on|off` - Toggle text-to-speech
- `/voice <mode>` - Voice input modes:
  - `off` - Disable voice input
  - `full` - Continuous listening, interrupts TTS on speech detection
  - `wait` - Pause listening while speaking (recommended, reduces self-interruption)
  - `stop` - Only stop on 'stop' keyword (planned)
  - `ptt` - Push-to-talk mode (planned)
- `/speed <number>` - Set TTS speed (0.5-2.0, default: 1.0, **pitch preserved**)
- `/tts_model <model>` - Switch TTS model:
  - `vits` - **Best quality** (requires espeak-ng)
  - `fast_pitch` - Good quality (works everywhere)
  - `glow-tts` - Alternative (similar quality to fast_pitch)
  - `tacotron2-DDC` - Legacy (slower, lower quality)
- `/whisper <model>` - Switch Whisper model (tiny|base|small|medium|large)
- `/stop` - Stop voice mode or TTS playback
- `/pause` - Pause current TTS playback (can be resumed)
- `/resume` - Resume paused TTS playback

**LLM Configuration:**
- `/model <name>` - Change LLM model (e.g., `/model gemma3:latest`)
- `/system <prompt>` - Set system prompt (e.g., `/system You are a helpful coding assistant`)
- `/temperature <val>` - Set temperature (0.0-2.0, default: 0.7)
- `/max_tokens <num>` - Set max tokens (default: 4096)

**Chat Management:**
- `/save <filename>` - Save chat history (e.g., `/save conversation`)
- `/load <filename>` - Load chat history (e.g., `/load conversation`)
- `/tokens` - Display token usage statistics

**Sending Messages:**
- `<message>` - Any text without `/` prefix is sent to the LLM

**Note**: Commands without `/` (except `stop`) are sent to the LLM as regular messages.

### Web API

```bash
# Run the web API example
abstractvoice-cli web

# With different host and port
abstractvoice-cli web --host 0.0.0.0 --port 8000
```

You can also run a simplified version that doesn't load the full models:

```bash
# Run the web API with simulation mode
abstractvoice-cli web --simulate
```

#### Troubleshooting Web API

If you encounter issues with the web API:

1. **404 Not Found**: Make sure you're accessing the correct endpoints (e.g., `/api/test`, `/api/tts`)
2. **Connection Issues**: Ensure no other service is using the port
3. **Model Loading Errors**: Try running with `--simulate` flag to test without loading models
4. **Dependencies**: Ensure all required packages are installed:
   ```bash
   pip install flask soundfile numpy requests
   ```
5. **Test with a simple Flask script**:
   ```python
   from flask import Flask
   app = Flask(__name__)
   @app.route('/')
   def home():
       return "Flask works!"
   app.run(host='127.0.0.1', port=5000)
   ```

### Simple Demo

```bash
# Run the simple example
abstractvoice-cli simple
```

## Documentation

### 📚 Documentation Overview

- **[README.md](README.md)** - This file: User guide, API reference, and examples
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Contribution guidelines and development setup
- **[CHANGELOG.md](CHANGELOG.md)** - Version history and release notes
- **[docs/](docs/)** - Technical documentation for developers

### 🎯 Quick Navigation

- **Getting Started**: [Installation](#installation) and [Quick Start](#quick-start)
- **Pause/Resume Control**: [TTS Control](#quick-reference-tts-control) section
- **Integration Examples**: [Integration Guide](#integration-guide-for-third-party-applications)
- **Technical Details**: [docs/architecture.md](docs/architecture.md) - How immediate pause/resume works
- **Development**: [CONTRIBUTING.md](CONTRIBUTING.md) - Setup and guidelines

## Component Overview

### VoiceManager

The main class that coordinates TTS and STT functionality:

```python
from abstractvoice import VoiceManager

# Simple initialization (automatic model selection)
# - Uses VITS if espeak-ng is installed (best quality)
# - Falls back to fast_pitch if espeak-ng is missing
manager = VoiceManager()

# Or specify a model explicitly
manager = VoiceManager(
    tts_model="tts_models/en/ljspeech/vits",  # Best quality (needs espeak-ng)
    # tts_model="tts_models/en/ljspeech/fast_pitch",  # Good (works everywhere)
    whisper_model="tiny",
    debug_mode=False
)

# === TTS (Text-to-Speech) ===

# Basic speech synthesis
manager.speak("Hello world")

# With speed control (pitch preserved via time-stretching!)
manager.speak("This is 20% faster", speed=1.2)
manager.speak("This is half speed", speed=0.5)

# Check if speaking
if manager.is_speaking():
    manager.stop_speaking()

# Pause and resume TTS (IMMEDIATE response)
manager.speak("This is a long sentence that can be paused and resumed immediately")
time.sleep(1)
success = manager.pause_speaking()  # Pause IMMEDIATELY (~20ms response)
if success:
    print("TTS paused immediately")

time.sleep(2)
success = manager.resume_speaking()  # Resume IMMEDIATELY from exact position
if success:
    print("TTS resumed from exact position")

# Check pause status
if manager.is_paused():
    manager.resume_speaking()

# Change TTS speed globally
manager.set_speed(1.3)  # All subsequent speech will be 30% faster

# Change TTS model dynamically
manager.set_tts_model("tts_models/en/ljspeech/glow-tts")

# Available TTS models (quality ranking):
# - "tts_models/en/ljspeech/vits" (BEST quality, requires espeak-ng)
# - "tts_models/en/ljspeech/fast_pitch" (fallback, works everywhere)
# - "tts_models/en/ljspeech/glow-tts" (alternative fallback)
# - "tts_models/en/ljspeech/tacotron2-DDC" (legacy)

# === STT (Speech-to-Text) ===

def on_transcription(text):
    print(f"You said: {text}")

manager.listen(on_transcription, on_stop=None)
manager.stop_listening()
manager.is_listening()

# Change Whisper model
manager.set_whisper("base")  # tiny, base, small, medium, large

# === Voice Modes ===

# Control how voice recognition behaves during TTS
manager.set_voice_mode("wait")  # Pause listening while speaking (recommended)
manager.set_voice_mode("full")  # Keep listening, interrupt on speech
manager.set_voice_mode("off")   # Disable voice recognition

# === VAD (Voice Activity Detection) ===

manager.change_vad_aggressiveness(2)  # 0-3, higher = more aggressive

# === Cleanup ===

manager.cleanup()
```

### TTSEngine

Handles text-to-speech synthesis:

```python
from abstractvoice.tts import TTSEngine

# Initialize with fast_pitch model (default, no external dependencies)
tts = TTSEngine(
    model_name="tts_models/en/ljspeech/fast_pitch",
    debug_mode=False,
    streaming=True  # Enable progressive playback for long text
)

# Speak with speed control (pitch preserved via time-stretching)
tts.speak(text, speed=1.2, callback=None)  # 20% faster, same pitch

# Immediate pause and resume control
success = tts.pause()      # Pause IMMEDIATELY (~20ms response)
success = tts.resume()     # Resume IMMEDIATELY from exact position
is_paused = tts.is_paused()  # Check if currently paused

tts.stop()       # Stop completely (cannot resume)
tts.is_active()  # Check if active
```

**Important Note on Speed Parameter:**
- The speed parameter now uses proper time-stretching (via librosa)
- Changing speed does NOT affect pitch anymore
- Range: 0.5 (half speed) to 2.0 (double speed)
- Example: `speed=1.3` makes speech 30% faster while preserving natural pitch

### VoiceRecognizer

Manages speech recognition with VAD:

```python
from abstractvoice.recognition import VoiceRecognizer

def on_transcription(text):
    print(f"Transcribed: {text}")

def on_stop():
    print("Stop command detected")

recognizer = VoiceRecognizer(transcription_callback=on_transcription,
                           stop_callback=on_stop, 
                           whisper_model="tiny",
                           debug_mode=False)
recognizer.start(tts_interrupt_callback=None)
recognizer.stop()
recognizer.change_whisper_model("base")
recognizer.change_vad_aggressiveness(2)
```

## Quick Reference: TTS Control

### Pause and Resume TTS

**Professional-grade pause/resume control** with immediate response and no terminal interference.

**In CLI/REPL:**
```bash
/pause    # Pause current TTS playback IMMEDIATELY
/resume   # Resume paused TTS playback IMMEDIATELY  
/stop     # Stop TTS completely (cannot resume)
```

**Programmatic Usage:**

#### Basic Pause/Resume
```python
from abstractvoice import VoiceManager
import time

vm = VoiceManager()

# Start speech
vm.speak("This is a long sentence that demonstrates immediate pause and resume functionality.")

# Pause immediately (takes effect within ~20ms)
time.sleep(1)
result = vm.pause_speaking()
if result:
    print("✓ TTS paused immediately")

# Resume immediately (takes effect within ~20ms)  
time.sleep(2)
result = vm.resume_speaking()
if result:
    print("✓ TTS resumed immediately")
```

#### Advanced Control with Status Checking
```python
from abstractvoice import VoiceManager
import time

vm = VoiceManager()

# Start long speech
vm.speak("This is a very long text that will be used to demonstrate the advanced pause and resume control features.")

# Wait and pause
time.sleep(1.5)
if vm.is_speaking():
    vm.pause_speaking()
    print("Speech paused")

# Check pause status
if vm.is_paused():
    print("Confirmed: TTS is paused")
    time.sleep(2)
    
    # Resume from exact position
    vm.resume_speaking()
    print("Speech resumed from exact position")

# Wait for completion
while vm.is_speaking():
    time.sleep(0.1)
print("Speech completed")
```

#### Interactive Control Example
```python
from abstractvoice import VoiceManager
import threading
import time

vm = VoiceManager()

def control_speech():
    """Interactive control in separate thread"""
    time.sleep(2)
    print("Pausing speech...")
    vm.pause_speaking()
    
    time.sleep(3)
    print("Resuming speech...")
    vm.resume_speaking()

# Start long speech
long_text = """
This is a comprehensive demonstration of AbstractVoice's immediate pause and resume functionality.
The system uses non-blocking audio streaming with callback-based control.
You can pause and resume at any time with immediate response.
The audio continues from the exact position where it was paused.
"""

# Start control thread
control_thread = threading.Thread(target=control_speech, daemon=True)
control_thread.start()

# Start speech (non-blocking)
vm.speak(long_text)

# Wait for completion
while vm.is_speaking() or vm.is_paused():
    time.sleep(0.1)

vm.cleanup()
```

#### Error Handling
```python
from abstractvoice import VoiceManager

vm = VoiceManager()

# Start speech
vm.speak("Testing pause/resume with error handling")

# Safe pause with error handling
try:
    if vm.is_speaking():
        success = vm.pause_speaking()
        if success:
            print("Successfully paused")
        else:
            print("No active speech to pause")
    
    # Safe resume with error handling
    if vm.is_paused():
        success = vm.resume_speaking()
        if success:
            print("Successfully resumed")
        else:
            print("Was not paused or playback completed")
            
except Exception as e:
    print(f"Error controlling TTS: {e}")
```

**Key Features:**
- **⚡ Immediate Response**: Pause/resume takes effect within ~20ms
- **🎯 Exact Position**: Resumes from precise audio position (no repetition)
- **🖥️ No Terminal Interference**: Uses OutputStream callbacks, never blocks terminal
- **🔒 Thread-Safe**: Safe to call from any thread or callback
- **📊 Reliable Status**: `is_paused()` and `is_speaking()` always accurate
- **🔄 Seamless Streaming**: Works with ongoing text synthesis

**How it works:**
- Uses `sounddevice.OutputStream` with callback function
- Pause immediately outputs silence in next audio callback (~20ms)
- Resume immediately continues audio output from exact position
- No blocking `sd.stop()` calls that interfere with terminal I/O
- Thread-safe with proper locking mechanisms

## Quick Reference: Speed & Model Control

### Changing TTS Speed

**In CLI/REPL:**
```bash
/speed 1.2    # 20% faster, pitch preserved
/speed 0.8    # 20% slower, pitch preserved
```

**Programmatically:**
```python
from abstractvoice import VoiceManager

vm = VoiceManager()

# Method 1: Set global speed
vm.set_speed(1.3)  # All speech will be 30% faster
vm.speak("This will be 30% faster")

# Method 2: Per-speech speed
vm.speak("This is 50% faster", speed=1.5)
vm.speak("This is normal speed", speed=1.0)
vm.speak("This is half speed", speed=0.5)

# Get current speed
current = vm.get_speed()  # Returns 1.3 from set_speed() above
```

### Changing TTS Model

**In CLI/REPL:**
```bash
/tts_model vits           # Best quality (needs espeak-ng)
/tts_model fast_pitch     # Good quality (works everywhere)
/tts_model glow-tts       # Alternative model
/tts_model tacotron2-DDC  # Legacy model
```

**Programmatically:**
```python
from abstractvoice import VoiceManager

# Method 1: Set at initialization
vm = VoiceManager(tts_model="tts_models/en/ljspeech/glow-tts")

# Method 2: Change dynamically at runtime
vm.set_tts_model("tts_models/en/ljspeech/fast_pitch")
vm.speak("Using fast_pitch now")

vm.set_tts_model("tts_models/en/ljspeech/glow-tts")
vm.speak("Using glow-tts now")

# Available models (quality ranking):
models = [
    "tts_models/en/ljspeech/vits",          # BEST (requires espeak-ng)
    "tts_models/en/ljspeech/fast_pitch",    # Good (works everywhere)
    "tts_models/en/ljspeech/glow-tts",      # Alternative fallback
    "tts_models/en/ljspeech/tacotron2-DDC"  # Legacy
]
```

### Complete Example: Experiment with Settings

```python
from abstractvoice import VoiceManager
import time

vm = VoiceManager()

# Test different models (vits requires espeak-ng)
for model in ["vits", "fast_pitch", "glow-tts", "tacotron2-DDC"]:
    full_name = f"tts_models/en/ljspeech/{model}"
    vm.set_tts_model(full_name)
    
    # Test different speeds with each model
    for speed in [0.8, 1.0, 1.2]:
        vm.speak(f"Testing {model} at {speed}x speed", speed=speed)
        while vm.is_speaking():
            time.sleep(0.1)
```

## Integration Guide for Third-Party Applications

AbstractVoice is designed as a lightweight, modular library for easy integration into your applications. This guide covers everything you need to know.

### Quick Start: Basic Integration

```python
from abstractvoice import VoiceManager

# 1. Initialize (automatic best-quality model selection)
vm = VoiceManager()

# 2. Text-to-Speech
vm.speak("Hello from my app!")

# 3. Speech-to-Text with callback
def handle_speech(text):
    print(f"User said: {text}")
    # Process text in your app...

vm.listen(on_transcription=handle_speech)
```

### Model Selection: Automatic vs Explicit

**Automatic (Recommended):**
```python
# Automatically uses best available model
vm = VoiceManager()
# → Uses VITS if espeak-ng installed (best quality)
# → Falls back to fast_pitch if espeak-ng missing
```

**Explicit:**
```python
# Force a specific model (bypasses auto-detection)
vm = VoiceManager(tts_model="tts_models/en/ljspeech/fast_pitch")

# Or change dynamically at runtime
vm.set_tts_model("tts_models/en/ljspeech/vits")
```

### Voice Quality Levels

| Model | Quality | Speed | Requirements |
|-------|---------|-------|--------------|
| **vits** | ⭐⭐⭐⭐⭐ Excellent | Fast | espeak-ng |
| **fast_pitch** | ⭐⭐⭐ Good | Fast | None |
| **glow-tts** | ⭐⭐⭐ Good | Fast | None |
| **tacotron2-DDC** | ⭐⭐ Fair | Slow | None |

### Customization Options

```python
from abstractvoice import VoiceManager

vm = VoiceManager(
    # TTS Configuration
    tts_model="tts_models/en/ljspeech/vits",  # Model to use
    
    # STT Configuration  
    whisper_model="base",  # tiny, base, small, medium, large
    
    # Debugging
    debug_mode=True  # Enable detailed logging
)

# Runtime customization
vm.set_speed(1.2)                    # Adjust TTS speed (0.5-2.0)
vm.set_tts_model("...")              # Change TTS model
vm.set_whisper("small")              # Change STT model
vm.set_voice_mode("wait")            # wait, full, or off
vm.change_vad_aggressiveness(2)      # VAD sensitivity (0-3)
```

### Integration Patterns

#### Pattern 1: TTS Only (No Voice Input)
```python
vm = VoiceManager()

# Speak with different speeds
vm.speak("Normal speed")
vm.speak("Fast speech", speed=1.5)
vm.speak("Slow speech", speed=0.7)

# Control playback with immediate response
if vm.is_speaking():
    success = vm.pause_speaking()  # Pause IMMEDIATELY (~20ms)
    if success:
        print("Speech paused immediately")
    # or
    vm.stop_speaking()   # Stop completely (cannot resume)

# Resume from exact position
if vm.is_paused():
    success = vm.resume_speaking()  # Resume IMMEDIATELY (~20ms)
    if success:
        print("Speech resumed from exact position")
```

#### Pattern 2: STT Only (No Text-to-Speech)
```python
vm = VoiceManager()

def process_speech(text):
    # Send to your backend, save to DB, etc.
    your_app.process(text)

vm.listen(on_transcription=process_speech)
```

#### Pattern 3: Full Voice Interaction
```python
vm = VoiceManager()

def on_speech(text):
    response = your_llm.generate(text)
    vm.speak(response)

def on_stop():
    print("User said stop")
    vm.cleanup()

vm.listen(
    on_transcription=on_speech,
    on_stop=on_stop
)
```

### Error Handling

```python
try:
    vm = VoiceManager()
    vm.speak("Test")
except Exception as e:
    print(f"TTS Error: {e}")
    # Handle missing dependencies, etc.

# Check model availability
try:
    vm.set_tts_model("tts_models/en/ljspeech/vits")
    print("VITS available")
except:
    print("VITS not available, using fallback")
    vm.set_tts_model("tts_models/en/ljspeech/fast_pitch")
```

### Threading and Async Support

AbstractVoice handles threading internally for TTS and STT:

```python
# TTS is non-blocking
vm.speak("Long text...")  # Returns immediately
# Your code continues while speech plays

# Check status
if vm.is_speaking():
    print("Still speaking...")

# Wait for completion
while vm.is_speaking():
    time.sleep(0.1)

# STT runs in background thread
vm.listen(on_transcription=callback)  # Returns immediately
# Callbacks fire on background thread
```

### Cleanup and Resource Management

```python
# Always cleanup when done
vm.cleanup()

# Or use context manager pattern
from contextlib import contextmanager

@contextmanager
def voice_manager():
    vm = VoiceManager()
    try:
        yield vm
    finally:
        vm.cleanup()

# Usage
with voice_manager() as vm:
    vm.speak("Hello")
```

### Configuration for Different Environments

**Development (fast iteration):**
```python
vm = VoiceManager(
    tts_model="tts_models/en/ljspeech/fast_pitch",  # Fast
    whisper_model="tiny",  # Fast STT
    debug_mode=True
)
```

**Production (best quality):**
```python
vm = VoiceManager(
    tts_model="tts_models/en/ljspeech/vits",  # Best quality
    whisper_model="base",  # Good accuracy
    debug_mode=False
)
```

**Embedded/Resource-Constrained:**
```python
vm = VoiceManager(
    tts_model="tts_models/en/ljspeech/fast_pitch",  # Lower memory
    whisper_model="tiny",  # Smallest model
    debug_mode=False
)
```

## Integration with Text Generation Systems

AbstractVoice is designed to be a lightweight, modular library that you can easily integrate into your own applications. Here are complete examples for common use cases:

### Example 1: Voice-Enabled Chatbot with Ollama

```python
from abstractvoice import VoiceManager
import requests
import time

# Initialize voice manager
voice_manager = VoiceManager()

# Function to call Ollama API
def generate_text(prompt):
    response = requests.post("http://localhost:11434/api/chat", json={
        "model": "granite3.3:2b",
        "messages": [{"role": "user", "content": prompt}],
        "stream": False
    })
    return response.json()["message"]["content"]

# Callback for speech recognition
def on_transcription(text):
    if text.lower() == "stop":
        return
        
    print(f"User: {text}")
    
    # Generate response
    response = generate_text(text)
    print(f"AI: {response}")
    
    # Speak response
    voice_manager.speak(response)

# Start listening
voice_manager.listen(on_transcription)

# Keep running until interrupted
try:
    while voice_manager.is_listening():
        time.sleep(0.1)
except KeyboardInterrupt:
    voice_manager.cleanup()
```

### Example 2: Voice-Enabled Assistant with OpenAI

```python
from abstractvoice import VoiceManager
import openai
import time

# Initialize
voice_manager = VoiceManager()
openai.api_key = "your-api-key"

def on_transcription(text):
    print(f"User: {text}")
    
    # Get response from OpenAI
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": text}]
    )
    
    ai_response = response.choices[0].message.content
    print(f"AI: {ai_response}")
    
    # Speak the response
    voice_manager.speak(ai_response)

# Start voice interaction
voice_manager.listen(on_transcription)

# Keep running
try:
    while voice_manager.is_listening():
        time.sleep(0.1)
except KeyboardInterrupt:
    voice_manager.cleanup()
```

### Example 3: Text-to-Speech Only (No Voice Input)

```python
from abstractvoice import VoiceManager
import time

# Initialize voice manager
voice_manager = VoiceManager()

# Simple text-to-speech
voice_manager.speak("Hello! This is a test of the text to speech system.")

# Wait for speech to finish
while voice_manager.is_speaking():
    time.sleep(0.1)

# Adjust speed
voice_manager.set_speed(1.5)
voice_manager.speak("This speech is 50% faster.")

while voice_manager.is_speaking():
    time.sleep(0.1)

# Cleanup
voice_manager.cleanup()
```

### Example 4: Speech-to-Text Only (No TTS)

```python
from abstractvoice import VoiceManager
import time

voice_manager = VoiceManager()

def on_transcription(text):
    print(f"Transcribed: {text}")
    # Do something with the transcribed text
    # e.g., save to file, send to API, etc.

# Start listening
voice_manager.listen(on_transcription)

# Keep running
try:
    while voice_manager.is_listening():
        time.sleep(0.1)
except KeyboardInterrupt:
    voice_manager.cleanup()
```

### Key Integration Points

**VoiceManager Configuration:**
```python
# Full configuration example
voice_manager = VoiceManager(
    tts_model="tts_models/en/ljspeech/fast_pitch",  # Default (no external deps)
    whisper_model="base",  # Whisper STT model (tiny, base, small, medium, large)
    debug_mode=True  # Enable debug logging
)

# Alternative TTS models (all pure Python, cross-platform):
# - "tts_models/en/ljspeech/fast_pitch" - Default (fast, good quality)
# - "tts_models/en/ljspeech/glow-tts" - Alternative (similar quality)
# - "tts_models/en/ljspeech/tacotron2-DDC" - Legacy (older, slower)

# Set voice mode (full, wait, off)
voice_manager.set_voice_mode("wait")  # Recommended to avoid self-interruption

# Adjust settings (speed now preserves pitch!)
voice_manager.set_speed(1.2)  # TTS speed (default is 1.0, range 0.5-2.0)
voice_manager.change_vad_aggressiveness(2)  # VAD sensitivity (0-3)
```

**Callback Functions:**
```python
def on_transcription(text):
    """Called when speech is transcribed"""
    print(f"User said: {text}")
    # Your custom logic here

def on_stop():
    """Called when user says 'stop'"""
    print("Stopping voice mode")
    # Your cleanup logic here

voice_manager.listen(
    on_transcription=on_transcription,
    on_stop=on_stop
)
```

## Perspectives

This is a test project that I designed with examples to work with Ollama, but I will adapt the examples and abstractvoice to work with any LLM provider (anthropic, openai, etc).

Next iteration will leverage directly [AbstractCore](https://www.abstractcore.ai) to handle everything related to LLM, their providers, models and configurations.

## License and Acknowledgments

AbstractVoice is licensed under the [MIT License](LICENSE).

This project depends on several open-source libraries and models, each with their own licenses. Please see [ACKNOWLEDGMENTS.md](ACKNOWLEDGMENTS.md) for a detailed list of dependencies and their respective licenses.

Some dependencies, particularly certain TTS models, may have non-commercial use restrictions. If you plan to use AbstractVoice in a commercial application, please ensure you are using models that permit commercial use or obtain appropriate licenses. 