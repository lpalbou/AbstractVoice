# Installation

AbstractVoice aims to work out of the box with:

- **TTS (default)**: Piper (ONNX; no system deps)
- **STT (default)**: faster-whisper (CTranslate2)
- **Audio I/O**: `sounddevice` (PortAudio) + `soundfile` + `webrtcvad`

## Install

```bash
pip install abstractvoice
```

## Optional extras

```bash
pip install "abstractvoice[cloning]"   # OpenF5-based cloning (heavy)
pip install "abstractvoice[chroma]"    # Chroma-4B (very heavy; torch/transformers)
pip install "abstractvoice[aec]"       # Optional echo cancellation (true barge-in)
pip install "abstractvoice[audio-fx]"  # Speed change without pitch change (librosa)
pip install "abstractvoice[stt]"       # Legacy openai-whisper + tiktoken (token stats)
```

## Offline-first model downloads

The REPL (`python -m abstractvoice cli`) runs with `allow_downloads=False` and will **not**
download weights implicitly. Prefetch explicitly:

```bash
# Piper voice model (per language). Cache: ~/.piper/models
python -m abstractvoice download --piper en

# STT model (faster-whisper). Cache: ~/.cache/huggingface by default
python -m abstractvoice download --stt small

# Voice cloning artifacts
python -m abstractvoice download --openf5
python -m abstractvoice download --chroma
```

## Audio device setup (common issues)

AbstractVoice uses **PortAudio** via `sounddevice`.

### macOS

- Ensure your terminal/IDE has **Microphone** permission (System Settings → Privacy & Security → Microphone).
- If audio devices fail to open, PortAudio can be installed with:

```bash
brew install portaudio
```

### Linux (Debian/Ubuntu)

```bash
sudo apt-get update
sudo apt-get install -y portaudio19-dev
```

### Windows

Usually works out of the box. If device access fails, check OS microphone permissions and installed audio drivers.

## Troubleshooting

- **Piper model not available locally**: run `python -m abstractvoice download --piper <lang>`.
- **Cloning runtime not ready**: run `/cloning_status` then `/cloning_download f5_tts|chroma` in the REPL (or use `python -m abstractvoice download ...`).
- **LLM API not reachable**: if you use the default Ollama endpoint, start it with `ollama serve`, or point the REPL at a different `--api`.

