## Task 004: Public API v0.6 (Clean Integrator Surface)

**Status**: ✅ Completed  
**Completed**: 2026-01-23  
**Priority**: P0

---

## What was delivered

### 1) Documented integrator contract

- Added/updated:
  - `docs/public_api.md`
  - `docs/overview.md`
  - `README.md` (short “happy path”)

The public API now explicitly includes:
- `speak()`, `speak_to_bytes()`, `speak_to_file()`
- `transcribe_file()`, `transcribe_from_bytes()`
- `listen()`, `stop_listening()`, `pause_listening()`, `resume_listening()`
- playback control: `pause_speaking()/resume_speaking()/stop_speaking()`
- stop phrase: **"ok stop"** / **"okay stop"**

### 2) Default engine strategy

- Default TTS: Piper
- Default STT: faster-whisper

Legacy engines are treated as optional/advanced.

