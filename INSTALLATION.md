# AbstractVoice Installation Guide

## Quick Start

```bash
# Complete installation (recommended)
pip install abstractvoice[all]

# Check if everything is working
python -m abstractvoice check-deps
```

## Installation Options

### 1. Core Package Only (Minimal)
```bash
pip install abstractvoice
```
- **What you get**: numpy, requests (2 dependencies)
- **Use case**: When you want to handle dependencies manually
- **Note**: TTS/STT functionality will give helpful error messages

### 2. Voice Functionality Groups

#### Complete Voice Features (Most Popular)
```bash
pip install abstractvoice[voice-full]
```
- **What you get**: TTS + STT + Audio I/O (all voice features)
- **Dependencies**: All PyTorch ecosystem with compatible versions
- **Use case**: Full voice assistant functionality

#### TTS Only (Lightweight)
```bash
pip install abstractvoice[core-tts]
```
- **What you get**: Text-to-speech only
- **Dependencies**: coqui-tts, PyTorch ecosystem
- **Use case**: Speaking without listening

#### STT Only (Lightweight)
```bash
pip install abstractvoice[core-stt]
```
- **What you get**: Speech-to-text only
- **Dependencies**: OpenAI Whisper, tiktoken
- **Use case**: Transcription without synthesis

#### Audio Processing Only
```bash
pip install abstractvoice[audio-only]
```
- **What you get**: Audio I/O and processing
- **Dependencies**: sounddevice, webrtcvad, PyAudio, soundfile
- **Use case**: Custom TTS/STT with AbstractVoice audio handling

### 3. Legacy Options (Still Available)
```bash
pip install abstractvoice[tts]    # TTS functionality
pip install abstractvoice[stt]    # STT functionality
pip install abstractvoice[web]    # Web API support
pip install abstractvoice[all]    # Everything
```

### 4. Development Installation
```bash
git clone https://github.com/lpalbou/abstractvoice.git
cd abstractvoice
pip install -e ".[all,dev]"
```

## Dependency Compatibility

### PyTorch Ecosystem (Critical for TTS)

AbstractVoice uses **tested, compatible versions** to avoid conflicts:

- **PyTorch**: 2.0.0 - 2.3.x (pinned to avoid conflicts)
- **TorchVision**: 0.15.0 - 0.18.x (explicitly included)
- **TorchAudio**: 2.0.0 - 2.3.x (matches PyTorch)
- **Coqui-TTS**: 0.27.0 - 0.29.x (latest compatible)

### Why Version Pinning Matters

The error `RuntimeError: operator torchvision::nms does not exist` occurs when:
- PyTorch and TorchVision versions are incompatible
- TorchVision is missing (not explicitly installed)
- Conda and pip installations conflict

**Our solution**: Install known-compatible versions automatically.

## Platform-Specific Setup

### macOS
```bash
# Install system audio dependencies (optional, for premium voices)
brew install espeak-ng

# Install AbstractVoice
pip install abstractvoice[voice-full]
```

### Linux (Ubuntu/Debian)
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install espeak-ng portaudio19-dev

# Install AbstractVoice
pip install abstractvoice[voice-full]
```

### Windows
```bash
# Windows usually works out of the box
pip install abstractvoice[voice-full]

# If you get audio errors, try:
conda install pyaudio
pip install abstractvoice[voice-full]
```

## Environment Management

### Recommended: Conda + Pip
```bash
# Create clean environment
conda create -n abstractvoice python=3.10
conda activate abstractvoice

# Install AbstractVoice with pip (not conda)
pip install abstractvoice[voice-full]
```

### Alternative: Pure Pip with Virtual Environment
```bash
python -m venv abstractvoice-env
source abstractvoice-env/bin/activate  # On Windows: abstractvoice-env\Scripts\activate
pip install abstractvoice[voice-full]
```

## GPU Support (Optional)

### CUDA 11.8 (Most Compatible)
```bash
# Install PyTorch with CUDA first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Then install AbstractVoice without PyTorch
pip install abstractvoice[core-stt,audio-only]
pip install coqui-tts librosa
```

### CUDA 12.x
```bash
# Check CUDA version
nvidia-smi

# Install compatible PyTorch (example for CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install AbstractVoice
pip install abstractvoice[voice-full]
```

## Troubleshooting

### Check Dependencies
```bash
# Run comprehensive dependency check
python -m abstractvoice check-deps
```

This will show:
- ✅ Installed and compatible packages
- ❌ Missing or incompatible packages
- ⚠️ Detected conflicts
- 💡 Specific fix recommendations

### Common Issues

#### 1. PyTorch/TorchVision Conflicts
**Error**: `RuntimeError: operator torchvision::nms does not exist`

**Solution**:
```bash
# Complete reinstall with compatible versions
pip uninstall torch torchvision torchaudio transformers
pip install abstractvoice[all]
```

#### 2. Audio Device Issues
**Error**: `OSError: PortAudio library not found` or `sounddevice` errors

**Solution**:
```bash
# macOS
brew install portaudio

# Linux
sudo apt-get install portaudio19-dev

# Windows (use conda)
conda install pyaudio
```

#### 3. Conda/Pip Conflicts
**Error**: Multiple PyTorch installations or version conflicts

**Solution**:
```bash
# Clean conda environment
conda create -n abstractvoice-clean python=3.10
conda activate abstractvoice-clean
pip install abstractvoice[voice-full]
```

#### 4. Import Errors After Installation
**Error**: `ImportError` or `ModuleNotFoundError`

**Solution**:
```bash
# Check what's actually installed
python -m abstractvoice check-deps

# Reinstall missing components
pip install abstractvoice[voice-full] --force-reinstall
```

## Testing Your Installation

### 1. Quick Test
```bash
python -c "from abstractvoice import VoiceManager; print('✅ AbstractVoice imported successfully')"
```

### 2. Full Functionality Test
```bash
python -m abstractvoice simple
```

### 3. Check Dependencies
```bash
python -m abstractvoice check-deps
```

### 4. Test TTS Only
```python
from abstractvoice import VoiceManager
vm = VoiceManager()
vm.speak("Hello! AbstractVoice is working correctly.")
```

## Model Download Information

### First Use Download
- **Size**: ~500MB per language (~2GB for all languages)
- **Location**: `~/.cache/huggingface/` (automatic)
- **Offline**: Works completely offline after first download
- **Languages**: English, French, Spanish, German, Italian

### Model Selection
AbstractVoice automatically selects the best model for your system:
1. **VITS** (premium quality) if espeak-ng is available
2. **Tacotron2-DDC** (reliable) as fallback
3. **Fast-pitch** (fastest) for older systems

## Performance Notes

- **Model Loading**: 2-5 seconds (cached after first use)
- **Memory Usage**: ~500MB-2GB depending on models loaded
- **CPU Usage**: Moderate (GPU optional for speed)
- **Pause/Resume**: <20ms latency

## Version Compatibility Matrix

| AbstractVoice | PyTorch | TorchVision | Python |
|---------------|---------|-------------|--------|
| 0.2.0+        | 2.0-2.3 | 0.15-0.18   | 3.8+   |

## Getting Help

1. **Check dependencies**: `python -m abstractvoice check-deps`
2. **Read error messages**: AbstractVoice provides specific installation instructions for each error
3. **GitHub Issues**: [Report problems](https://github.com/lpalbou/abstractvoice/issues)
4. **Documentation**: Check the [README](README.md) for usage examples

---

**Next Steps**: After installation, see [README.md](README.md) for usage examples and [llms-full.txt](llms-full.txt) for AI integration guides.