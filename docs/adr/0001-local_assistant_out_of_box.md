## ADR 0001: Local Voice Assistant Works Out-of-the-Box

**Date**: 2026-01-23  
**Status**: Accepted

---

## Context

AbstractVoice must support a local “voice assistant” experience via the `abstractvoice` command:

- user speaks (microphone capture)
- assistant transcribes (STT)
- assistant answers (LLM)
- assistant speaks (TTS)

This must work on a fresh machine with **no extra system installs** beyond `pip install abstractvoice`.

Historically we relied on `PyAudio`, which is a frequent installation failure point across platforms.

---

## Decision

- Use **sounddevice** (PortAudio) for microphone capture in-process.
- Use **webrtcvad** for low-latency VAD (pip-installable).
- Use **faster-whisper** for STT (core dependency).
- Keep the legacy STT path (openai-whisper) optional only.

---

## Consequences

### Positive

- `listen()` can work on the base install (no `PyAudio` required).
- Unified audio backend for playback + capture (PortAudio).

### Negative / Risks

- PortAudio availability can still be an issue on some Linux setups.
  - Mitigation: document common system packages in `docs/installation.md`.

