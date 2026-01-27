# Getting Started (Developer & Local Testing)

## Setup (recommended)

Create and use a local virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run the test suite:

```bash
pytest -q
```

## Run the local voice assistant

```bash
python -m abstractvoice cli --debug
python -m abstractvoice cli --verbose
```

The REPL is **offline-first** (no implicit model downloads). Prefetch what you need:

```bash
python -m abstractvoice download --piper en
python -m abstractvoice download --stt small
```

## Common checks

- **TTS**: use `/speak hello` in the REPL
- **STT from file**: use `/transcribe path/to/audio.wav`
- **Stop phrase**: say **"ok stop"** while listening to stop voice mode safely
