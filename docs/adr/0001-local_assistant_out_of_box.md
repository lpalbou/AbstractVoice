## ADR 0001: Local Voice Assistant Works Out-of-the-Box

**Date**: 2026-01-23  
**Status**: Superseded for the base install; platform/granular profiles retained

---

## Context

AbstractVoice must support a local “voice assistant” experience via the `abstractvoice` command:

- user speaks (microphone capture)
- assistant transcribes (STT)
- assistant answers (LLM)
- assistant speaks (TTS)

Original constraint: this should work on a fresh machine with **no extra system installs** beyond
`pip install abstractvoice`.

Historically we relied on `PyAudio`, which is a frequent installation failure point across platforms.

2026-05-08 refinement for AbstractVoice `0.9.1`: the base install is
remote-first. Bare `abstractvoice` / `VoiceManager()` selects OpenAI remote
TTS/STT and requires `OPENAI_API_KEY` or `remote_api_key=...`. This ADR now
applies only to explicit local assistant installs such as
`pip install "abstractvoice[apple]"`, `pip install "abstractvoice[gpu]"`, or
granular compositions like `pip install "abstractvoice[supertonic,stt,audio-io]"`,
plus explicit local engine selection.

---

## Decision

- Use **sounddevice** (PortAudio) for microphone capture in-process.
- Use **webrtcvad** for low-latency VAD (pip-installable).
- Use **faster-whisper** for local STT in the local assistant profile.
- Remove the legacy OpenAI Whisper fallback from the public install contract.
- Keep local assistant UX available through platform or granular extras and
  explicit local engines.
- Default direct `VoiceManager()` and AbstractCore/Gateway capability-plugin
  usage to remote OpenAI/OpenAI-compatible TTS/STT.

---

## Consequences

### Positive

- `listen()` can work in the local assistant profile (no `PyAudio` required).
- Unified audio backend for playback + capture (PortAudio).
- Bare `abstractvoice` stays remote-light and aligns with AbstractCore/Gateway
  deployments.

### Negative / Risks

- PortAudio availability can still be an issue on some Linux setups.
  - Mitigation: document common system packages in `README.md` and `docs/installation.md`.
- Existing callers that relied on implicit local `auto` behavior must select
  local engines explicitly and install either a platform profile or the needed
  granular extras.

## Related

- Framework ADR 0033: install profiles, config entry points, and server boundaries.
- Framework ADR 0028: capabilities plugins and library/framework modes.
- AbstractVoice backlog: `docs/backlog/completed/037_lightweight_openai_compatible_packaging.md`
