## Task 015: ADR 0002 Phase 1 hardening (wait mode default + stop phrase semantics)

**Status**: Completed  
**Priority**: P0  

---

## Summary

Aligned Phase 1 behavior with ADR 0002 while keeping the implementation **robust without AEC** and keeping the API surface small.

Key decision: in `wait` mode we do **not** fully stop mic capture during TTS, because that would make the stop phrase unusable. Instead, we **suppress normal transcriptions** and **disable “interrupt on any speech”** during TTS, while keeping stop phrase detection active.

---

## Changes

- **Default voice mode**: `VoiceManager` now defaults to `_voice_mode = "wait"`.
- **Stop phrase semantics**:
  - Stop phrase triggers `stop_speaking()` (immediate playback stop).
  - Stop phrase does **not** forcibly stop listening; integrators can decide that in `on_stop`.
- **Conservative matching**:
  - Removed acceptance of plain `"stop"` (too risky with speaker echo).
  - Added dependency-free helper `abstractvoice/stop_phrase.py` and used it in recognition.
- **During TTS** (`wait/stop/ptt`):
  - Disable interrupt-on-speech (`pause_tts_interrupt()`).
  - Suppress normal transcriptions (`pause_transcriptions()` / `resume_transcriptions()`).

---

## Tests

- Added `tests/test_adr0002_phase1.py`:
  - default mode is `wait`
  - stop phrase stops speaking without forcing stop_listening
  - conservative stop phrase normalization (no `webrtcvad` dependency)

All tests: **29 passed, 1 skipped**.

