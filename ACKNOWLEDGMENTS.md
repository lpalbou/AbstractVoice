# Acknowledgments

AbstractVoice stands on top of excellent open-source software and openly released model work. Thank you to all maintainers and contributors.

For licensing caveats around **model weights** and **voice files** (which are separate assets from this MIT-licensed library), read `docs/voices-and-licenses.md`.

## Core dependencies (shipped by default)

- Piper / `piper-tts` (local neural TTS): https://github.com/rhasspy/piper
- faster-whisper (STT): https://github.com/SYSTRAN/faster-whisper
- CTranslate2 (inference runtime used by faster-whisper): https://github.com/OpenNMT/CTranslate2
- Hugging Face Hub (`huggingface_hub`) (artifact downloads): https://github.com/huggingface/huggingface_hub
- SoundDevice + PortAudio (audio I/O): https://github.com/spatialaudio/python-sounddevice and http://www.portaudio.com/
- SoundFile (WAV/FLAC/OGG I/O): https://github.com/bastibe/python-soundfile
- WebRTC VAD (`webrtcvad`) (voice activity detection): https://github.com/wiseman/py-webrtcvad
- NumPy: https://github.com/numpy/numpy
- Requests: https://github.com/psf/requests
- appdirs: https://github.com/ActiveState/appdirs

## Optional features (extras)

These are **opt-in** via extras in `pyproject.toml` (see `docs/installation.md`):

- F5-TTS (`abstractvoice[cloning]`) for cloning backends: https://github.com/SWivid/F5-TTS
- Chroma runtime deps (`abstractvoice[chroma]`) for Chroma-4B inference: https://huggingface.co/FlashLabs/Chroma-4B
  - PyTorch: https://github.com/pytorch/pytorch
  - Transformers: https://github.com/huggingface/transformers
- AEC (`abstractvoice[aec]`) for true barge-in on speakers: https://github.com/shichaog/AEC-Audio-Processing
- Audio effects (`abstractvoice[audio-fx]`): https://github.com/librosa/librosa
- Legacy Whisper + token stats (`abstractvoice[stt]`): https://github.com/openai/whisper and https://github.com/openai/tiktoken
- Web demo/API (`abstractvoice[web]`): https://github.com/pallets/flask

## Models and voices

AbstractVoice may download model weights and voice files at runtime (explicitly or on demand, depending on `allow_downloads`).

- Piper voices are cached under `~/.piper/models` (see `abstractvoice/adapters/tts_piper.py`).
- faster-whisper models use the Hugging Face cache by default (see `abstractvoice/adapters/stt_faster_whisper.py`).
- Cloning artifacts are cached under `~/.cache/abstractvoice/*` (see `abstractvoice/cloning/engine_f5.py` and `abstractvoice/cloning/engine_chroma.py`).

Always verify upstream licenses/usage terms for the specific models/voices you deploy or redistribute. See `docs/voices-and-licenses.md`.
