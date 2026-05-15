# Installation

AbstractVoice has a lightweight remote-first base install plus explicit local extras:

- **Python**: `>=3.9` (see `pyproject.toml`)
- **Base install**: OpenAI/OpenAI-compatible TTS/STT/profile/clone adapters and AbstractCore plugin discovery
- **Platform/local extras**: Piper TTS, Supertonic 3 ONNX TTS, faster-whisper STT, audio I/O, AEC where supported, and local cloning/TTS engines

## Install

```bash
pip install abstractvoice
```

This is the remote/plugin-friendly install. `VoiceManager()` defaults to OpenAI
hosted audio; set `OPENAI_API_KEY` or pass `remote_api_key=...`. The CLI/web
examples are install-aware: with the plain install they start on OpenAI, while
`abstractvoice[all-apple]` and `abstractvoice[all-gpu]` start on Supertonic by
default. For local desktop/REPL voice and local cloning engines, prefer the
profile for your machine:

```bash
pip install "abstractvoice[apple]"     # Apple Silicon / native macOS
pip install "abstractvoice[gpu]"       # NVIDIA/AMD GPU-oriented local stack
pip install "abstractvoice[all-apple]" # Apple stack + local web example
pip install "abstractvoice[all-gpu]"   # GPU stack + local web example
```

For OpenAI-compatible HTTP audio endpoints, install AbstractVoice beside
AbstractCore Server and configure a remote provider or install local voice
runtimes explicitly:

```bash
pip install "abstractcore[server]" abstractvoice
OPENAI_API_KEY=... python -m abstractcore.server.app
```

AbstractCore provides `/v1/audio/speech` and `/v1/audio/transcriptions`;
AbstractVoice is discovered as the voice/audio capability backend.

## Optional extras

```bash
pip install "abstractvoice[apple]"     # Apple Silicon local stack: Piper, Supertonic, faster-whisper, AEC, cloning/TTS engines
pip install "abstractvoice[gpu]"       # GPU local stack: Piper, Supertonic, faster-whisper, AEC, cloning/TTS engines
pip install "abstractvoice[all-apple]" # Apple stack + local FastAPI browser example
pip install "abstractvoice[all-gpu]"   # GPU stack + local FastAPI browser example
pip install "abstractvoice[piper]"     # Local Piper TTS only
pip install "abstractvoice[supertonic]" # Local Supertonic 3 ONNX TTS only
pip install "abstractvoice[stt]"       # Local faster-whisper STT
pip install "abstractvoice[audio-io]"  # Microphone/playback/VAD dependencies
pip install "abstractvoice[cloning]"   # explicit OpenF5-based cloning (heavy; Python 3.10+)
pip install "abstractvoice[chroma]"    # Chroma-4B (very heavy; torch/transformers)
pip install "abstractvoice[audiodit]"  # LongCat-AudioDiT (heavy; torch/transformers)
pip install "abstractvoice[omnivoice]" # recommended/default local cloning + OmniVoice TTS/design (very heavy; Python 3.10+)
pip install "abstractvoice[openai]"    # Hosted OpenAI intent extra (no extra deps today)
pip install "abstractvoice[openai-compatible]" # Generic compatible provider intent extra
pip install "abstractvoice[aec]"       # Optional echo cancellation (true barge-in)
pip install "abstractvoice[audio-fx]"  # Speed change without pitch change (librosa)
pip install "abstractvoice[web]"       # Local FastAPI browser example
pip install "abstractvoice[web,supertonic]" # Web example + Supertonic dependency
pip install "abstractvoice[web,omnivoice]" # Web example + OmniVoice dependency
```

`abstractvoice[web]` intentionally stays lightweight: it installs the browser
server, but no local engines. Compose it with `abstractvoice[all-apple]`,
`abstractvoice[all-gpu]`, or granular engine extras such as
`abstractvoice[supertonic]` / `abstractvoice[omnivoice]` for smaller installs.

Remote OpenAI-compatible audio:

```bash
# OpenAI hosted audio
export OPENAI_API_KEY=...
python - <<'PY'
from abstractvoice import VoiceManager
vm = VoiceManager(tts_engine="openai", stt_engine="openai")
vm.set_profile("nova")  # OpenAI voice id
wav = vm.speak_to_bytes("Hello from remote TTS.", format="wav")
PY

# Any OpenAI-compatible /v1 server, including AbstractCore Server
export ABSTRACTVOICE_REMOTE_BASE_URL=http://localhost:8000/v1
python - <<'PY'
from abstractvoice import VoiceManager
vm = VoiceManager(tts_engine="openai-compatible", stt_engine="openai-compatible")
print([p.profile_id for p in vm.get_profiles()])
wav = vm.speak_to_bytes("Hello through a compatible endpoint.", format="wav")
PY
```

Remote cloning is provider-specific. For compatible services, configure
`ABSTRACTVOICE_REMOTE_BASE_URL` and use `cloning_engine="openai-compatible"`;
the default custom clone route is `POST /voice/clone` and should return
`{"voice_id": "..."}` or `{"id": "..."}`.
For `cloning_engine="openai"`, OpenAI custom voice creation is org-gated and
requires explicit consent configuration such as
`ABSTRACTVOICE_OPENAI_VOICE_CONSENT_ID`.

Remote profile/voice listing is part of the AbstractVoice-compatible extension
contract. Compatible servers can expose `GET /v1/audio/voices`; the adapter
calls it as `GET /audio/voices` relative to `remote_base_url`, parses
`profiles`, `voices`, `cloned_voices`, or OpenAI-style `data`, and exposes the
ids through `VoiceManager.get_profiles()`. Static ids can be configured with
`ABSTRACTVOICE_REMOTE_TTS_VOICES`.

Python-version notes:

- Python 3.9 supports the lightweight base, local Piper/faster-whisper extras,
  local Supertonic 3 ONNX TTS, the web example, and AudioDiT TTS/prompt-audio cloning.
- Supertonic is the recommended local base TTS path. OmniVoice is the
  recommended/default local cloning backend.
- OpenF5/F5-TTS, Chroma, and OmniVoice require Python 3.10+ because their
  upstream runtimes do.
- AEC requires Python 3.11+ because `aec-audio-processing` declares that floor.

Note (OmniVoice): OmniVoice uses the torch/torchaudio/torchvision stack. If you
already have an incompatible `torchvision` installed (common after changing
torch-backed extras), you may see import errors like:

- `RuntimeError: operator torchvision::nms does not exist`

Fix by installing a torchvision build that matches your torch version. For the
torch 2.8 family this is typically:

```bash
python -m pip install --upgrade --force-reinstall "torchvision==0.23.*"
```

## Offline-first model downloads

The REPL (`python -m abstractvoice cli`) runs with `allow_downloads=False` and will **not**
download weights implicitly. Prefetch explicitly:

```bash
# Piper voice model (per language). Cache: ~/.piper/models
python -m abstractvoice download --piper en

# Supertonic 3 ONNX TTS (weights + 10 built-in voice styles).
# Cache: ~/.cache/abstractvoice/supertonic-3
python -m abstractvoice download --supertonic

# STT model (faster-whisper). Cache: ~/.cache/huggingface by default
python -m abstractvoice download --stt small

# Voice cloning artifacts (optional; require extras)
pip install "abstractvoice[cloning]"   # for --openf5 (Python 3.10+)
python -m abstractvoice download --openf5

pip install "abstractvoice[chroma]"    # for --chroma (GPU-heavy)
python -m abstractvoice download --chroma

pip install "abstractvoice[audiodit]"  # for --audiodit (LongCat-AudioDiT-1B)
python -m abstractvoice download --audiodit

pip install "abstractvoice[omnivoice]" # for --omnivoice (OmniVoice)
python -m abstractvoice download --omnivoice
```

The same operations are available via the convenience entrypoint:

```bash
abstractvoice-prefetch --piper en
abstractvoice-prefetch --supertonic
abstractvoice-prefetch --stt small
abstractvoice-prefetch --openf5
abstractvoice-prefetch --chroma
abstractvoice-prefetch --audiodit
abstractvoice-prefetch --omnivoice
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
- **Supertonic model not available locally**: run `python -m abstractvoice download --supertonic`.
- **Cloning runtime not ready**: for the default OmniVoice cloning path, run `/cloning_status` then `/cloning_download omnivoice` in the REPL (or use `python -m abstractvoice download --omnivoice`). Use `/cloning_download f5_tts|chroma|audiodit` only when selecting those engines explicitly.
- **LLM API not reachable (REPL only)**: the default provider preset is Ollama at `http://localhost:11434` (OpenAI-compatible `POST /v1/chat/completions`). Start it with `ollama serve`, or point the REPL at a different `--provider`/`--api` base URL.
- **Sanity-check your environment**: run `python -m abstractvoice check-deps` (or `abstractvoice check-deps`) to print a dependency report.

See also: `docs/faq.md`.
