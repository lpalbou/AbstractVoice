## Task 015: ADR 0002 Phase 1 hardening (wait mode default + stop phrase semantics)

**Status**: Completed  
**Priority**: P0  

---

## Summary

Aligned Phase 1 behavior with ADR 0002 while keeping the implementation **robust without AEC** and keeping the API surface small.

Amendment (2026-02-04): the shipped behavior distinguishes:
- `wait` mode: **pause mic processing** during TTS (strict turn-taking; no voice barge-in while speaking)
- `stop` mode: keep mic processing running, but during TTS **suppress normal transcriptions** and keep a rolling stop-phrase detector active (so “ok stop” can cut playback)

Implementation lives in `abstractvoice/vm/core.py` (TTS callbacks) and `abstractvoice/recognition.py` (stop-phrase detection).

---

## Changes

- **Default voice mode**: `VoiceManager` now defaults to `_voice_mode = "wait"`.
- **Stop phrase semantics**:
  - Stop phrase triggers `stop_speaking()` (immediate playback stop).
  - Stop phrase does **not** forcibly stop listening; integrators can decide that in `on_stop`.
- **Conservative matching**:
  - Added dependency-free helper `abstractvoice/stop_phrase.py` and used it in recognition.
  - Kept bare `"stop"` available, but made it conservative (confirmation required in the rolling detector path to reduce false triggers from speaker echo).
- **During TTS** (voice modes; see `abstractvoice/vm/core.py`):
  - `wait`: `pause_listening()` / `resume_listening()`
  - `stop` / `ptt`: `pause_tts_interrupt()` + `pause_transcriptions()` (stop phrase still works via rolling detector)
  - `full`: no suppression (speech can interrupt TTS; best with AEC/headset)

---

## Tests

- Added `tests/test_adr0002_phase1.py`:
  - default mode is `wait`
  - stop phrase stops speaking without forcing `stop_listening()`
  - conservative stop phrase normalization (no `webrtcvad` dependency)
- Added `tests/test_voice_mode_wait_pauses_listening_during_tts.py`:
  - `wait` mode pauses mic processing during TTS start and resumes on end

All tests: **29 passed, 1 skipped**.
