## Task 003: VoiceManager Adapter Integration Hardening

**Status**: ✅ Completed  
**Completed**: 2026-01-23  
**Priority**: P0  

---

## Goal

Stabilize the system-level integration of the new default engines:
- Piper (TTS)
- faster-whisper (STT)

So that the library and REPL work reliably as a local assistant and for integrators.

---

## What was implemented

### 1) Stable TTS control surface for adapter engines

- Added `abstractvoice/tts/adapter_tts_engine.py`
  - TTSEngine-compatible wrapper around `TTSAdapter`
  - Uses existing `NonBlockingAudioPlayer`
  - Preserves `stop/pause/resume/is_active/is_paused` and callback wiring

### 2) VoiceManager transcription path uses faster-whisper by default

- Updated `VoiceManager.transcribe_file()` to prefer the faster-whisper adapter when available.
- Kept legacy `openai-whisper` as optional fallback only.

### 3) Process-exit and audio teardown stability

- Hardened PortAudio stream teardown (abort/stop/close)
- Added best-effort cleanup hooks to avoid rare crash-on-exit scenarios.

### 4) REPL stability improvements

- REPL now respects `/tts off` for test messages.
- Added `/transcribe <path>` command to validate STT via `transcribe_file()`.

---

## Verification

Full test suite passes:
- `pytest -q` → **26 passed, 1 skipped**

---

## Notes

This task intentionally optimized for a clean “works out-of-the-box” experience rather than preserving legacy behavior.

