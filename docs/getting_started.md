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
abstractvoice cli --debug
```

## Common checks

- **TTS**: use `/speak hello` in the REPL
- **STT from file**: use `/transcribe path/to/audio.wav`
- **Stop phrase**: say **"ok stop"** while listening to stop voice mode safely

