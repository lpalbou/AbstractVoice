# Getting started

This is the recommended next step after `README.md`.

If you want the supported integrator contract, see `docs/public_api.md`.

## Requirements

- Python `>=3.8` (see `pyproject.toml`)
- For microphone input: OS-level microphone permissions for your terminal/IDE

## Install

```bash
pip install abstractvoice
```

Optional extras are documented in `docs/installation.md` (cloning / Chroma / AEC / audio-fx / legacy STT).

## 60-second smoke test (no mic required)

Start the REPL:

```bash
abstractvoice --verbose
```

From a source checkout (without installing the console script), use:
```bash
python -m abstractvoice cli --verbose
```

In the REPL, run:

- `/speak hello` (tests TTS without calling an LLM)

If Piper can’t speak in offline-first mode, prefetch a voice model:

```bash
abstractvoice-prefetch --piper en
```

## Minimal library usage

```python
from abstractvoice import VoiceManager

vm = VoiceManager()
vm.speak("Hello from AbstractVoice.")
```

The public entry point is `abstractvoice.VoiceManager` (`abstractvoice/voice_manager.py`).

## Enable microphone input (voice modes)

By default, the REPL does **not** start microphone capture. Enable it explicitly:

```bash
abstractvoice --voice-mode stop
```

From a source checkout:
```bash
python -m abstractvoice cli --voice-mode stop
```

Recommended modes (implemented in `abstractvoice/vm/core.py` and `abstractvoice/recognition.py`):

- `stop` (recommended on speakers): keeps listening; during TTS it suppresses normal transcriptions but still lets you say “ok stop” to cut playback.
- `wait` (strict turn-taking): pauses mic processing while speaking.
- `full` (barge-in by speech): best with AEC or a headset; speakers can self-interrupt.

See `docs/repl_guide.md` for commands and `docs/adr/0002_barge_in_interruption.md` for rationale.

## Offline-first prefetch (recommended for deployments)

The REPL runs with `allow_downloads=False`, so prefetch explicitly:

```bash
abstractvoice-prefetch --stt small
abstractvoice-prefetch --piper en
```

For cloning engines (optional / large), see `docs/installation.md` and `docs/voices-and-licenses.md`.

## Contributing / local development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest -q
```

Internal notes: `docs/development.md`.
