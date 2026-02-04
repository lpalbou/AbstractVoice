# Overview

AbstractVoice is a Python library for **voice I/O** around AI applications:

- **TTS (text → audio)**: default engine is **Piper** (cross‑platform, no system deps).
- **STT (audio → text)**: default engine is **faster‑whisper** (fast, multilingual).

The main entry point for integrators is `abstractvoice.VoiceManager`.

See `docs/acronyms.md` for acronyms used in documentation.

Next reads:
- `docs/getting-started.md` (recommended setup + first smoke tests)
- `docs/installation.md` (platform notes + optional extras)
- `docs/public_api.md` (supported integrator contract)
- `docs/repl_guide.md` (end-to-end validation via CLI)

## What “clean” means in this project

- **One obvious happy path** for end users and third‑party integrators.
- **Headless‑friendly** primitives (bytes/file) for client/server architectures.
- **Small public API**, stable semantics, and clear errors.
- Internals are modular and replaceable via adapters.

## Common usage modes

### 1) Local app (desktop)

- `speak()` plays to speakers
- `pause_speaking()/resume_speaking()` control playback

### 2) Backend server (headless)

- `speak_to_bytes()` / `speak_to_file()` to send audio to clients
- `transcribe_from_bytes()` / `transcribe_file()` for uploaded audio

### 3) Manual validation (REPL)

- `python -m abstractvoice cli` is the fastest end‑to‑end smoke test.
  - The REPL is offline-first: no implicit model downloads.
