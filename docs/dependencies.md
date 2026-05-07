# Dependencies (direct) + models/weights

This document lists **direct** dependencies declared in `pyproject.toml`, plus the **models/weights** AbstractVoice may download.

Why direct-only?

- **Direct dependencies** (what we declare) are stable and reviewable.
- **Transitive dependencies** (what pip resolves) vary by platform and change over time; we call out only a few *notable runtimes* where it helps debugging.

If you need licensing guidance for model weights and voices, also read `docs/voices-and-licenses.md`.

---

## Lightweight base dependencies (installed by default)

These are always installed with `pip install abstractvoice`.

- **numpy**
  - **Why**: array operations and audio sample handling.
  - **Where**: widely used across `abstractvoice/*` (audio + cloning glue).
  - **Repo**: `https://github.com/numpy/numpy`
  - **License**: `https://github.com/numpy/numpy/blob/main/LICENSE.txt`

- **requests**
  - **Why**: HTTP calls (mainly integration / utility flows).
  - **Where**: various network helpers; also used by some upstream libs.
  - **Repo**: `https://github.com/psf/requests`
  - **License**: `https://github.com/psf/requests/blob/main/LICENSE`

- **appdirs**
  - **Why**: OS-appropriate cache/data directory locations.
  - **Where**: used in cache/storage helpers.
  - **Repo**: `https://github.com/ActiveState/appdirs`
  - **License**: `https://github.com/ActiveState/appdirs/blob/master/LICENSE.txt`

---

## Optional extras (opt-in)

### `abstractvoice[voice]` — local Piper + faster-whisper + audio I/O

This is the replacement for the old local-first base install.

- **piper-tts**
  - **Why**: local neural TTS backend (Piper-first).
  - **Where**: `abstractvoice/adapters/tts_piper.py`
  - **Repo**: `https://github.com/rhasspy/piper`
  - **License**: `https://github.com/rhasspy/piper/blob/master/LICENSE`

- **faster-whisper**
  - **Why**: local STT backend (fast Whisper inference).
  - **Where**: `abstractvoice/adapters/stt_faster_whisper.py`
  - **Repo**: `https://github.com/SYSTRAN/faster-whisper`
  - **License**: `https://github.com/SYSTRAN/faster-whisper/blob/master/LICENSE`

- **sounddevice**
  - **Why**: microphone/speaker I/O (PortAudio bindings).
  - **Where**: `abstractvoice/recognition.py`, `abstractvoice/tts/tts_engine.py`, `abstractvoice/audio/*`
  - **Repo**: `https://github.com/spatialaudio/python-sounddevice`
  - **License**: `https://github.com/spatialaudio/python-sounddevice/blob/master/LICENSE`

- **soundfile**
  - **Why**: WAV/FLAC/OGG read/write (libsndfile bindings).
  - **Where**: cloning store normalization, tests, file utilities.
  - **Repo**: `https://github.com/bastibe/python-soundfile`
  - **License**: `https://github.com/bastibe/python-soundfile/blob/master/LICENSE`

- **webrtcvad**
  - **Why**: VAD (voice activity detection) for `listen()` and voice modes.
  - **Where**: `abstractvoice/vad/voice_detector.py`, `abstractvoice/recognition.py`
  - **Repo**: `https://github.com/wiseman/py-webrtcvad`
  - **License**: `https://github.com/wiseman/py-webrtcvad/blob/master/LICENSE.txt`

### `abstractvoice[cloning]`, `[chroma]`, `[audiodit]`, `[omnivoice]` — model prefetch support

- **huggingface_hub**
  - **Why**: download and cache optional engine model weights/artifacts.
  - **Where**: `abstractvoice/cloning/engine_f5.py`, `abstractvoice/cloning/engine_chroma.py`, `abstractvoice/audiodit/runtime.py`, `abstractvoice/omnivoice/runtime.py`
  - **Repo**: `https://github.com/huggingface/huggingface_hub`
  - **License**: `https://github.com/huggingface/huggingface_hub/blob/main/LICENSE`

### `abstractvoice[audiodit]` — LongCat-AudioDiT (TTS + prompt-audio cloning)

Supported on Python 3.9+. Python 3.9 uses the latest compatible Transformers
4.x line; Python 3.10+ uses the Transformers 5.x line.

Python packages:

- **torch**
  - **Why**: tensor runtime for AudioDiT inference.
  - **Where**: `abstractvoice/audiodit/*`, `abstractvoice/cloning/engine_audiodit.py`
  - **Repo**: `https://github.com/pytorch/pytorch`
  - **License**: `https://github.com/pytorch/pytorch/blob/main/LICENSE`

- **transformers**
  - **Why**: HF `PreTrainedModel`/tokenizer APIs used by AudioDiT + text encoder.
  - **Where**: `abstractvoice/audiodit/*`, `abstractvoice/audiodit/runtime.py`
  - **Repo**: `https://github.com/huggingface/transformers`
  - **License**: `https://github.com/huggingface/transformers/blob/main/LICENSE`

- **einops**
  - **Why**: model block reshaping utilities (used by AudioDiT modules).
  - **Where**: `abstractvoice/audiodit/modeling_audiodit.py`
  - **Repo**: `https://github.com/arogozhnikov/einops`
  - **License**: `https://github.com/arogozhnikov/einops/blob/master/LICENSE`

- **sentencepiece**
  - **Why**: tokenizer runtime (used by UMT5 tokenizer).
  - **Where**: pulled by Transformers tokenizers depending on model.
  - **Repo**: `https://github.com/google/sentencepiece`
  - **License**: `https://github.com/google/sentencepiece/blob/master/LICENSE`

- **safetensors**
  - **Why**: weight format used by HF models.
  - **Where**: model load via Transformers.
  - **Repo**: `https://github.com/huggingface/safetensors`
  - **License**: `https://github.com/huggingface/safetensors/blob/main/LICENSE`

Models/weights (Hugging Face):

- **LongCat-AudioDiT-1B** (weights + model license statement)
  - **Model**: `https://huggingface.co/meituan-longcat/LongCat-AudioDiT-1B`
  - **License file**: `https://huggingface.co/meituan-longcat/LongCat-AudioDiT-1B/blob/main/LICENSE`
  - **Cache**: `~/.cache/huggingface` (default)
  - **Prefetch**: `python -m abstractvoice download --audiodit` or `abstractvoice-prefetch --audiodit`

- **UMT5 text encoder** (default: `google/umt5-base`)
  - **Model**: `https://huggingface.co/google/umt5-base`
  - **License**: `https://huggingface.co/google/umt5-base` (model card metadata; license is `apache-2.0`)
  - **Cache**: `~/.cache/huggingface` (default)

Vendored code:

- **LongCat-AudioDiT upstream repository (MIT)**
  - **Repo**: `https://github.com/meituan-longcat/LongCat-AudioDiT`
  - **License**: `https://github.com/meituan-longcat/LongCat-AudioDiT/blob/main/LICENSE`
  - **What we ship**: a HuggingFace-compatible derived implementation under `abstractvoice/audiodit/*` to avoid `trust_remote_code`.

### `abstractvoice[omnivoice]` — OmniVoice (omnilingual TTS + cloning / design)

Requires Python 3.10+ because the upstream `omnivoice` package declares that
Python floor.

Python packages:

- **omnivoice**
  - **Why**: OmniVoice engine runtime + model wrapper.
  - **Where**: `abstractvoice/omnivoice/*`, `abstractvoice/adapters/tts_omnivoice.py`, `abstractvoice/cloning/engine_omnivoice.py`
  - **Repo**: `https://github.com/k2-fsa/OmniVoice`
  - **License**: `https://github.com/k2-fsa/OmniVoice/blob/master/LICENSE`

- **torch / torchaudio / torchvision**
  - **Why**: tensor runtime for OmniVoice inference.
  - **Repo**: `https://github.com/pytorch/pytorch`
  - **License**: `https://github.com/pytorch/pytorch/blob/main/LICENSE`

- **transformers**
  - **Repo**: `https://github.com/huggingface/transformers`
  - **License**: `https://github.com/huggingface/transformers/blob/main/LICENSE`

- **accelerate**
  - **Repo**: `https://github.com/huggingface/accelerate`
  - **License**: `https://github.com/huggingface/accelerate/blob/main/LICENSE`

Models/weights (Hugging Face):

- **OmniVoice** (weights + model license statement)
  - **Model**: `https://huggingface.co/k2-fsa/OmniVoice`
  - **Cache**: `~/.cache/huggingface` (default)
  - **Prefetch**: `python -m abstractvoice download --omnivoice` or `abstractvoice-prefetch --omnivoice`

### `abstractvoice[chroma]` — Chroma-4B cloning (GPU-heavy)

Requires Python 3.10+ in AbstractVoice packaging because Chroma depends on
Transformers 5.x.

- **torch / torchaudio / torchvision**
  - **Repo**: `https://github.com/pytorch/pytorch`
  - **License**: `https://github.com/pytorch/pytorch/blob/main/LICENSE`

- **transformers**
  - **Repo**: `https://github.com/huggingface/transformers`
  - **License**: `https://github.com/huggingface/transformers/blob/main/LICENSE`

- **accelerate**
  - **Repo**: `https://github.com/huggingface/accelerate`
  - **License**: `https://github.com/huggingface/accelerate/blob/main/LICENSE`

- **av (PyAV)**
  - **Repo**: `https://github.com/PyAV-Org/PyAV`
  - **License**: `https://github.com/PyAV-Org/PyAV/blob/main/LICENSE.txt`

- **librosa / audioread / pillow / safetensors**
  - **librosa repo/license**: `https://github.com/librosa/librosa` / `https://github.com/librosa/librosa/blob/main/LICENSE.md`
  - **audioread repo/license**: `https://github.com/beetbox/audioread` / `https://github.com/beetbox/audioread/blob/master/LICENSE`
  - **pillow repo/license**: `https://github.com/python-pillow/Pillow` / `https://github.com/python-pillow/Pillow/blob/main/LICENSE`
  - **safetensors repo/license**: `https://github.com/huggingface/safetensors` / `https://github.com/huggingface/safetensors/blob/main/LICENSE`

Model:

- **Chroma-4B**
  - **Model**: `https://huggingface.co/FlashLabs/Chroma-4B`
  - **License**: see model card + repo files.

### `abstractvoice[cloning]` — OpenF5 / F5-TTS cloning

Requires Python 3.10+ because current upstream `f5-tts` packages import
Python 3.10-only annotations during inference. On Python 3.9, use
`abstractvoice[audiodit]` with the `audiodit` cloning engine for tested
prompt-audio cloning.

- **f5-tts**
  - **Why**: cloning engine (`f5_tts`) backend.
  - **Where**: `abstractvoice/cloning/engine_f5.py`
  - **Repo**: `https://github.com/SWivid/F5-TTS`
  - **License**: `https://github.com/SWivid/F5-TTS/blob/main/LICENSE`

### `abstractvoice[audio-fx]` — Audio effects

- **librosa**
  - **Why**: optional “speed change without pitch change”.
  - **Where**: used by the audio FX path; surfaced via `docs/installation.md`.
  - **Repo**: `https://github.com/librosa/librosa`
  - **License**: `https://github.com/librosa/librosa/blob/main/LICENSE.md`

### `abstractvoice[aec]` — Acoustic echo cancellation

Requires Python 3.11+ because `aec-audio-processing` declares that Python
floor.

- **aec-audio-processing**
  - **Why**: optional AEC for “full voice” barge-in on speakers.
  - **Where**: `abstractvoice/aec/*`
  - **Repo**: `https://github.com/shichaog/AEC-Audio-Processing`
  - **License**: `https://github.com/shichaog/AEC-Audio-Processing/blob/main/LICENSE`

### `abstractvoice[web]` — Local web example

The base web extra installs only the browser server stack. Use
`abstractvoice[web-cloning]`, `abstractvoice[web-audiodit]`,
`abstractvoice[web-omnivoice]`, `abstractvoice[web-chroma]`, or
`abstractvoice[web-full]` when you want a one-command install for the web UI
plus optional local voice engines.

- **FastAPI**
  - **Why**: local browser example routes for TTS, transcription, assistant/user voice selection, voice cloning upload forms, and discussion playback (`abstractvoice web`).
  - **Where**: `abstractvoice/examples/web_ui.py`
  - **Repo**: `https://github.com/fastapi/fastapi`
  - **License**: `https://github.com/fastapi/fastapi/blob/master/LICENSE`

- **Uvicorn**
  - **Why**: local ASGI server for `abstractvoice web`.
  - **Where**: `abstractvoice/examples/web_ui.py`
  - **Repo**: `https://github.com/encode/uvicorn`
  - **License**: `https://github.com/encode/uvicorn/blob/master/LICENSE.md`

- **python-multipart**
  - **Why**: parse browser/OpenAI-compatible audio upload forms for transcription and voice cloning.
  - **Where**: `abstractvoice/examples/web_ui.py`
  - **Repo**: `https://github.com/Kludex/python-multipart`
  - **License**: `https://github.com/Kludex/python-multipart/blob/master/LICENSE.txt`

### `abstractvoice[stt]` — Faster-Whisper STT

- **faster-whisper**
  - **Why**: local STT adapter path.
  - **Where**: `abstractvoice/adapters/stt_faster_whisper.py`
  - **Repo**: `https://github.com/SYSTRAN/faster-whisper`
  - **License**: `https://github.com/SYSTRAN/faster-whisper/blob/master/LICENSE`

### `abstractvoice[legacy-stt]` — Legacy Whisper + token stats

- **openai-whisper**
  - **Repo**: `https://github.com/openai/whisper`
  - **License**: `https://github.com/openai/whisper/blob/main/LICENSE`

- **tiktoken**
  - **Repo**: `https://github.com/openai/tiktoken`
  - **License**: `https://github.com/openai/tiktoken/blob/main/LICENSE`

---

## Notable transitive runtimes (debugging-oriented)

- **CTranslate2** (used by `faster-whisper`)
  - Repo: `https://github.com/OpenNMT/CTranslate2`
  - License: `https://github.com/OpenNMT/CTranslate2/blob/master/LICENSE`

- **ONNX Runtime** (used by Piper)
  - Repo: `https://github.com/microsoft/onnxruntime`
  - License: `https://github.com/microsoft/onnxruntime/blob/main/LICENSE`

---

## Development dependencies

These are used for local development only (not required at runtime).

- **pytest**: `https://github.com/pytest-dev/pytest` — `https://github.com/pytest-dev/pytest/blob/main/LICENSE`
- **black**: `https://github.com/psf/black` — `https://github.com/psf/black/blob/main/LICENSE`
- **flake8**: `https://github.com/PyCQA/flake8` — `https://github.com/PyCQA/flake8/blob/main/LICENSE`
