# AbstractVoice

A modular Python library for **voice I/O** around AI applications.

- **TTS (default)**: Piper (cross-platform, no system deps)
- **STT (default)**: faster-whisper
- **Local assistant**: `listen()` + `speak()` with playback/listening control
- **Headless/server**: `speak_to_bytes()` / `speak_to_file()` and `transcribe_*`

Status: **alpha** (`0.6.3`). The supported integrator surface is documented in `docs/api.md`.

Next: `docs/getting-started.md` (recommended setup + first smoke tests).

## AbstractFramework ecosystem

AbstractVoice is part of the **AbstractFramework** ecosystem:

- AbstractFramework (umbrella): https://github.com/lpalbou/AbstractFramework
- AbstractCore (agents/capabilities): https://github.com/lpalbou/abstractcore
- AbstractRuntime (runtime + artifacts): https://github.com/lpalbou/abstractruntime

Integration points (code evidence):

- AbstractCore capability plugin entry point: `pyproject.toml` → `[project.entry-points."abstractcore.capabilities_plugins"]`  
  Implementation: `abstractvoice/integrations/abstractcore_plugin.py`
- AbstractRuntime ArtifactStore adapter (optional, duck-typed): `abstractvoice/artifacts.py`

**Important**: AbstractVoice is a **voice I/O library** (TTS/STT + optional cloning). It is not an agent framework and it does not implement an LLM server.
In the AbstractFramework stack, **AbstractCore** is the intended place to run agents and expose OpenAI-compatible HTTP endpoints; AbstractVoice is discovered as a **capability backend plugin** and provides the TTS/STT implementation.

```mermaid
flowchart LR
  App[Your app / REPL] --> VM[abstractvoice.VoiceManager]
  VM --> TTS[Piper (TTS)]
  VM --> STT[faster-whisper (STT)]
  VM --> IO[sounddevice / PortAudio]

  subgraph AbstractFramework
    AC[AbstractCore] -. capability plugin .-> VM
    AR[AbstractRuntime] -. optional ArtifactStore .-> VM
  end
```

The shipped AbstractCore integration is via the capability plugin above. The `abstractvoice` REPL is a **demonstrator/smoke-test harness** (see `docs/repl_guide.md`) and includes a minimal OpenAI-compatible LLM HTTP client (`abstractvoice/examples/llm_provider.py`) for convenience.

---

## Install

Requires Python `>=3.10` (see `pyproject.toml`).

```bash
pip install abstractvoice
```

Optional extras (feature flags):

```bash
pip install "abstractvoice[all]"
```

Notes:
- `abstractvoice[all]` enables most optional features (incl. cloning + AEC + audio-fx), but **does not** include the GPU-heavy Chroma runtime or AudioDiT.
- For the full list of extras (and platform troubleshooting), see `docs/installation.md`.

### Explicit model downloads (recommended; never implicit in the REPL)

Some features rely on large model weights/artifacts. AbstractVoice will **not**
download these implicitly inside the REPL (offline-first).

After installing, prefetch explicitly (cross-platform).

Recommended (most users):

```bash
abstractvoice-prefetch --piper en
abstractvoice-prefetch --stt small
```

Optional (voice cloning artifacts):

```bash
pip install "abstractvoice[cloning]"
abstractvoice-prefetch --openf5

# Heavy (torch/transformers):
pip install "abstractvoice[audiodit]"
abstractvoice-prefetch --audiodit

# GPU-heavy:
pip install "abstractvoice[chroma]"
abstractvoice-prefetch --chroma
```

Equivalent `python -m` form:

```bash
python -m abstractvoice download --piper en
python -m abstractvoice download --stt small
python -m abstractvoice download --openf5   # optional; requires abstractvoice[cloning]
python -m abstractvoice download --chroma   # optional; requires abstractvoice[chroma] (GPU-heavy)
python -m abstractvoice download --audiodit # optional; requires abstractvoice[audiodit]
```

Notes:
- `--piper <lang>` downloads the Piper ONNX voice for that language into `~/.piper/models`.
- `--openf5` is ~5.4GB. `--chroma` is very large (GPU-heavy).

---

## Quick smoke tests

### REPL (fastest end-to-end)

```bash
abstractvoice --verbose
# or (from a source checkout):
python -m abstractvoice cli --verbose
```

Notes:
- Mic voice input is **off by default** for fast startup. Enable with `--voice-mode stop` (or in-session: `/voice stop`).
- The REPL is **offline-first**: no implicit model downloads. Use the explicit download commands above.
- The REPL is primarily a **demonstrator**. For production agent/server use in the AbstractFramework ecosystem, run AbstractCore and use AbstractVoice via its capability plugin (see `docs/api.md` → “Integrations”).

See `docs/repl_guide.md`.

### Minimal Python

```python
from abstractvoice import VoiceManager

vm = VoiceManager()
vm.speak("Hello! This is AbstractVoice.")
```

---

## Public API (stable surface)

See `docs/api.md` for the supported integrator contract.

At a glance:
- **TTS**: `speak()`, `stop_speaking()`, `pause_speaking()`, `resume_speaking()`, `speak_to_bytes()`, `speak_to_file()`
- **STT**: `transcribe_file()`, `transcribe_from_bytes()`
- **Mic**: `listen()`, `stop_listening()`, `pause_listening()`, `resume_listening()`

---

## Documentation (minimal set)

- **Docs index**: `docs/README.md`
- **Getting started**: `docs/getting-started.md`
- **FAQ**: `docs/faq.md`
- **Orientation**: `docs/overview.md`
- **Acronyms**: `docs/acronyms.md`
- **Public API**: `docs/api.md`
- **REPL guide**: `docs/repl_guide.md`
- **Install troubleshooting**: `docs/installation.md`
- **Multilingual support**: `docs/multilingual.md`
- **Architecture (internal)**: `docs/architecture.md` + `docs/adr/`
- **Model management (Piper-first)**: `docs/model-management.md`
- **Licensing notes**: `docs/voices-and-licenses.md`

---

## Project

- **Changelog**: `CHANGELOG.md`
- **Contributing**: `CONTRIBUTING.md`
- **Security**: `SECURITY.md`
- **Acknowledgments**: `ACKNOWLEDGMENTS.md`

## License

MIT. See `LICENSE`.
