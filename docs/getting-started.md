# Getting started

Start here after `README.md` when you want to confirm the package works, then
choose either the lightweight remote path or local/offline engines.

Use `docs/api.md` for the supported integrator contract, `docs/architecture.md`
for the implementation map, and `docs/faq.md` for cache/history reset and common
troubleshooting.

## Requirements

- Python `>=3.9` (see `pyproject.toml`)
- `OPENAI_API_KEY` for `VoiceManager()` defaults or explicit remote REPL/web TTS
- For microphone input: OS-level microphone permissions for your terminal/IDE

## Install

```bash
pip install abstractvoice
```

The plain install is lightweight and remote/plugin oriented. For fully local
inference, listening, Supertonic/Piper TTS, and cloning engines, install
`abstractvoice[apple]` or `abstractvoice[gpu]`, or compose granular extras such
as `abstractvoice[supertonic,stt,audio-io]`. Optional extras are documented in
`docs/installation.md`.

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

The interactive REPL default is `--tts-engine auto`: installed Supertonic first,
installed Piper second, then OpenAI remote. A plain lightweight install starts
on OpenAI; `abstractvoice[all-apple]` and `abstractvoice[all-gpu]` start on
Supertonic. If the status line says `openai (remote)`, set `OPENAI_API_KEY` or
switch to a local engine.

For local/offline TTS instead:

```bash
pip install "abstractvoice[supertonic]"
abstractvoice-prefetch --supertonic
abstractvoice --tts-engine supertonic --stt-engine faster_whisper --verbose
```

For the smaller Piper fallback:

```bash
pip install "abstractvoice[piper]"
abstractvoice-prefetch --piper en
abstractvoice --tts-engine piper --stt-engine faster_whisper --verbose
```

## Optional Browser Example

The local web UI is a small FastAPI example around `VoiceManager`: discussion
read-through with separate assistant/user voices, text to WAV, audio-file
transcription, and a tiny optional LLM dialogue panel for OpenAI-compatible
local providers such as Ollama. It is not the production server surface; use
AbstractCore Server for production OpenAI-compatible HTTP endpoints.

```bash
pip install "abstractvoice[web]"
abstractvoice web --port 5000
```

For a local web lab with Supertonic/faster-whisper:

```bash
pip install "abstractvoice[web,supertonic,stt]"
abstractvoice-prefetch --supertonic
abstractvoice web --port 5000 --tts-engine supertonic --stt-engine faster_whisper
```

For a remote OpenAI-compatible web smoke test:

```bash
abstractvoice web --port 5000 --tts-engine openai-compatible --stt-engine openai-compatible --remote-base-url http://localhost:8000/v1
```

If you want the browser UI and a smaller optional engine install, compose
extras directly, such as `abstractvoice[web,supertonic]` or
`abstractvoice[web,omnivoice]`.

Then open `http://127.0.0.1:5000`.

The web example is offline-first by default. Its `auto` TTS default has the same
interactive behavior as the REPL: installed Supertonic, installed Piper, then
remote OpenAI. Prefetch models first, or start it with `--allow-downloads` when
you explicitly want web requests to download missing models. The Voice panel
includes a base TTS engine selector; switching engines resets the base profile
to that engine/language default. Selecting a cloned voice can take a while on first use because the
cloning backend loads weights and builds prompt/runtime caches; the browser UI
shows a busy overlay while that work is happening.

The browser voice-cloning action validates a new clone by synthesizing a short
sample before reporting success. If an optional engine cannot load, the stored
clone is removed and the backend error is shown instead of leaving a broken
voice in the selector.

For the dialogue panel, start a compatible local LLM server separately (for
example Ollama on `http://localhost:11434`), choose a model in the page, then
use **Ask Assistant**. The browser owns the short chat history; the example
server only forwards one `/v1/chat/completions` request.

## Minimal library usage

```python
from abstractvoice import VoiceManager

vm = VoiceManager()
vm.speak("Hello from AbstractVoice.")
```

This uses OpenAI remote audio and reads `OPENAI_API_KEY`. For local inference:

```python
from abstractvoice import VoiceManager

vm = VoiceManager(tts_engine="supertonic", stt_engine="faster_whisper")
vm.speak("Hello from the local stack.")
```

For Supertonic fixed-profile local TTS, install `abstractvoice[supertonic]`,
prefetch with `abstractvoice-prefetch --supertonic`, and use
`VoiceManager(tts_engine="supertonic")`.

The public entry point is `abstractvoice.VoiceManager` (`abstractvoice/voice_manager.py`).

## Recommended in AbstractFramework: use AbstractCore

If you’re using AbstractVoice inside the AbstractFramework ecosystem, the intended architecture is:

- **AbstractCore** runs agents and exposes OpenAI-compatible endpoints.
- **AbstractVoice** is installed alongside it and provides TTS/STT as a **capability backend plugin**.

Pointers:

- `docs/api.md` → “Integrations (AbstractFramework ecosystem)”
- Capability plugin implementation: `abstractvoice/integrations/abstractcore_plugin.py`

If you’re integrating into the AbstractFramework ecosystem (AbstractCore / AbstractRuntime), see:
- `README.md` (ecosystem overview)
- `docs/api.md` (Integrations section; code pointers)

Minimal AbstractCore Server smoke test:

```bash
pip install "abstractcore[server]" abstractvoice
OPENAI_API_KEY=... python -m abstractcore.server.app

# TTS through AbstractCore + AbstractVoice
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"Hello from AbstractVoice through AbstractCore.","format":"wav"}' \
  --output hello.wav

# STT through AbstractCore + AbstractVoice
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@hello.wav" \
  -F "language=en"
```

If the server is configured with `ABSTRACTCORE_SERVER_API_KEY`, include
`Authorization: Bearer <key>` in those requests.

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
abstractvoice-prefetch --supertonic
```

For cloning engines (optional / large), see `docs/installation.md` and `docs/voices-and-licenses.md`.

Current engine caveats, including AudioDiT direct TTS quality and OmniVoice
profile stability, are tracked in `docs/known-issues.md`.

## Clear local history or caches

Inside the REPL, `/clear` resets the LLM message history sent to the provider.
`/reset` also resets active voice state. Saved memories only exist when you run
`/save <name>`.

Terminal command history, cloned voices, and model caches live in separate local
directories. The exact reset commands are in `docs/faq.md`.

## Contributing / local development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest -q
```

See also: `CONTRIBUTING.md`, `SECURITY.md`, and internal notes in `docs/development.md`.
