# Installation Guide

AbstractVoice is designed to work **out of the box** on all systems with automatic quality upgrades when possible.

## 🚀 Quick Start (Recommended)

### Step 1: Basic Installation
```bash
# Install AbstractVoice with all features (works everywhere)
pip install abstractvoice[all]
```

### Step 2: Verify Installation
```bash
# Test that everything works
python -c "from abstractvoice import VoiceManager; vm = VoiceManager(); print('✅ Installation successful!')"
```

### Step 3: Optional Quality Upgrade
```bash
# For better voice quality (if you want the absolute best)
# macOS:
brew install espeak-ng

# Linux:
sudo apt-get install espeak-ng

# Windows:
conda install espeak-ng
```

**That's it!** AbstractVoice will automatically use the best available models for your system.

## 📦 Installation Options

### Minimal Installation
```bash
# Just the core package (2 dependencies)
pip install abstractvoice

# Add features as needed
pip install abstractvoice[tts]      # Text-to-speech
pip install abstractvoice[stt]      # Speech-to-text
pip install abstractvoice[voice]    # Audio I/O
pip install abstractvoice[all]      # Everything (recommended)
```

### Language-Specific Installation
```bash
# Single language with all features
pip install abstractvoice[fr]       # French
pip install abstractvoice[es]       # Spanish
pip install abstractvoice[de]       # German
pip install abstractvoice[it]       # Italian
```

## Operating System Specific Instructions

### macOS

AbstractVoice works out of the box on macOS with minimal setup:

```bash
# Basic installation
pip install "abstractvoice[multilingual]"

# For best English quality (optional)
brew install espeak-ng

# Verify installation
python -c "from abstractvoice import VoiceManager; print('✅ Installation successful')"
```

#### macOS Troubleshooting

**Audio Permission Issues:**
```bash
# If you get audio permission errors, go to:
# System Preferences > Security & Privacy > Privacy > Microphone
# Add Terminal or your Python IDE to allowed apps
```

**Homebrew Missing:**
```bash
# Install Homebrew first
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Then install espeak-ng
brew install espeak-ng
```

### Linux (Ubuntu/Debian)

Linux requires additional system audio dependencies:

```bash
# Update package list
sudo apt-get update

# Install audio system dependencies
sudo apt-get install -y \
    portaudio19-dev \
    python3-pyaudio \
    alsa-utils \
    pulseaudio

# For enhanced voice quality (optional)
sudo apt-get install espeak-ng

# Install AbstractVoice
pip install "abstractvoice[multilingual]"

# Test audio system
aplay /usr/share/sounds/alsa/Front_Left.wav

# Verify installation
python3 -c "from abstractvoice import VoiceManager; print('✅ Installation successful')"
```

#### Linux Distribution Specific

**CentOS/RHEL/Fedora:**
```bash
# Install dependencies
sudo dnf install portaudio-devel python3-pyaudio alsa-utils

# Or for older versions
sudo yum install portaudio-devel python3-pyaudio alsa-utils

# Install espeak-ng (optional)
sudo dnf install espeak-ng

# Install AbstractVoice
pip install "abstractvoice[multilingual]"
```

**Arch Linux:**
```bash
# Install dependencies
sudo pacman -S portaudio python-pyaudio alsa-utils

# Install espeak-ng (optional)
sudo pacman -S espeak-ng

# Install AbstractVoice
pip install "abstractvoice[multilingual]"
```

#### Linux Troubleshooting

**Audio Issues:**
```bash
# Check audio devices
python -c "import sounddevice as sd; print(sd.query_devices())"

# Test audio output
speaker-test -t wav -c 2

# If no audio, restart audio services
sudo systemctl restart pulseaudio
# or
pulseaudio --kill && pulseaudio --start
```

**Permission Issues:**
```bash
# Add user to audio group
sudo usermod -a -G audio $USER

# Logout and login again, or use:
newgrp audio
```

### Windows

Windows installation requires more careful dependency management:

#### Option 1: Conda (Recommended)

```bash
# Install Miniconda or Anaconda first
# Download from: https://docs.conda.io/en/latest/miniconda.html

# Create new environment (recommended)
conda create -n abstractvoice python=3.9
conda activate abstractvoice

# Install audio dependencies
conda install pyaudio

# Install AbstractVoice
pip install "abstractvoice[multilingual]"

# Verify installation
python -c "from abstractvoice import VoiceManager; print('✅ Installation successful')"
```

#### Option 2: Pip with Build Tools

```bash
# Install Visual C++ Build Tools first
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Install dependencies
pip install pyaudio

# Install AbstractVoice
pip install "abstractvoice[multilingual]"
```

#### Option 3: Pre-compiled Wheels

```bash
# Download PyAudio wheel for your Python version from:
# https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio

# Install downloaded wheel
pip install PyAudio-0.2.11-cp39-cp39-win_amd64.whl

# Install AbstractVoice
pip install "abstractvoice[multilingual]"
```

#### Windows Troubleshooting

**Visual C++ Missing:**
```bash
# Download and install Microsoft C++ Build Tools
# URL: https://visualstudio.microsoft.com/visual-cpp-build-tools/
# Select: "C++ build tools" workload
```

**Audio Issues:**
```bash
# Check Windows audio services
# Win + R, type: services.msc
# Ensure "Windows Audio" service is running

# Test in Python
python -c "import sounddevice as sd; print(sd.query_devices())"
```

## Development Installation

For development or contributing to AbstractVoice:

```bash
# Clone repository
git clone https://github.com/lpalbou/abstractvoice.git
cd abstractvoice

# Install in development mode
pip install -e ".[dev,multilingual]"

# Run tests
pytest

# Check code quality
black .
flake8 .
```

## Docker Installation

For containerized environments:

```dockerfile
# Dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    python3-pyaudio \
    alsa-utils \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

# Install AbstractVoice
RUN pip install "abstractvoice[multilingual]"

# Copy your application
COPY . /app
WORKDIR /app

CMD ["python", "your_app.py"]
```

Build and run:
```bash
docker build -t abstractvoice-app .
docker run -it --device /dev/snd abstractvoice-app
```

## Virtual Environment Setup

### Using venv
```bash
# Create virtual environment
python -m venv abstractvoice-env

# Activate (Linux/macOS)
source abstractvoice-env/bin/activate

# Activate (Windows)
abstractvoice-env\Scripts\activate

# Install
pip install "abstractvoice[multilingual]"
```

### Using conda
```bash
# Create environment
conda create -n abstractvoice python=3.9

# Activate
conda activate abstractvoice

# Install dependencies
conda install pyaudio

# Install AbstractVoice
pip install "abstractvoice[multilingual]"
```

## Verification and Testing

After installation, verify everything works:

```python
#!/usr/bin/env python3
"""Installation verification script"""

import sys

def test_installation():
    print("🔧 Testing AbstractVoice installation...")

    # Test basic import
    try:
        from abstractvoice import VoiceManager
        print("✅ Import successful")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

    # Test audio system
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        print(f"✅ Audio system: {len(devices)} devices found")
    except Exception as e:
        print(f"⚠️ Audio system warning: {e}")

    # Test TTS models
    try:
        vm = VoiceManager(debug_mode=True)
        print("✅ English TTS model loaded")
    except Exception as e:
        print(f"❌ English TTS failed: {e}")
        return False

    # Test multilingual
    try:
        vm_fr = VoiceManager(language='fr', debug_mode=True)
        print("✅ French TTS model loaded")
    except Exception as e:
        print(f"⚠️ Multilingual TTS warning: {e}")

    # Test basic speech (optional)
    try:
        vm.speak("Installation test successful!")
        print("✅ Speech synthesis working")
    except Exception as e:
        print(f"⚠️ Speech synthesis warning: {e}")

    print("🎉 Installation verification complete!")
    return True

if __name__ == "__main__":
    success = test_installation()
    sys.exit(0 if success else 1)
```

Run the verification:
```bash
python verify_installation.py
```

## Common Issues and Solutions

### Issue: ModuleNotFoundError: No module named 'pyaudio'

**Solution:**
```bash
# macOS
brew install portaudio
pip install pyaudio

# Linux
sudo apt-get install portaudio19-dev
pip install pyaudio

# Windows
conda install pyaudio
```

### Issue: "espeak-ng not found"

**Solution:**
```bash
# This is just a warning for English TTS quality
# AbstractVoice will automatically fallback to fast_pitch model

# To fix (optional):
# macOS: brew install espeak-ng
# Linux: sudo apt-get install espeak-ng
# Windows: conda install espeak-ng
```

### Issue: "No audio devices found"

**Solution:**
```bash
# Check audio system
python -c "import sounddevice as sd; print(sd.query_devices())"

# Linux: restart audio
sudo systemctl restart pulseaudio

# Windows: check Windows Audio service
# macOS: check System Preferences > Sound
```

### Issue: Slow model loading

**Solution:**
```bash
# Models are downloaded once and cached
# First run may be slow (downloading ~1GB for XTTS-v2)
# Subsequent runs are fast

# Check cache location:
python -c "import torch; print(torch.hub.get_dir())"
```

## Performance Optimization

### Memory Usage
```python
# For memory-constrained environments
vm = VoiceManager(
    language='en',
    tts_model="tts_models/en/ljspeech/fast_pitch",  # Smaller model
    whisper_model="tiny"  # Smallest Whisper model
)
```

### Speed Optimization
```python
# Pre-load models for faster response
vm_en = VoiceManager(language='en')
vm_fr = VoiceManager(language='fr')
vm_es = VoiceManager(language='es')

# Use same instance for multiple speech calls
vm.speak("First text")
vm.speak("Second text")  # No model reload
```

This completes the comprehensive installation guide for all major platforms and use cases.