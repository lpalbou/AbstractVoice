## Task 020: Cloning ref_text auto-fallback (no user-provided transcript)

**Date**: 2026-01-23  
**Status**: Completed  
**Priority**: P0  

---

## Summary

When a cloned voice has no `reference_text`, we now generate it **once**, persist it in the voice store, and reuse it for all future synthesis. This prevents repeated re-transcription and reduces “poisoned prompt” artifacts across utterances.

---

## Changes

- `abstractvoice/cloning/manager.py`
  - Added `reference_text_whisper_model` (default `small`) for one-time ASR.
  - Added `_ensure_reference_text(voice_id)`:
    - if missing: transcribe (<=15s clip), normalize conservatively, persist.
    - (amended 2026-01-28) STT uses a 3-pass consensus strategy before persisting.
  - `speak_to_bytes()` and `speak_to_audio_chunks()` now call `_ensure_reference_text()` and pass the persisted text to the engine.

- `abstractvoice/cloning/store.py`
  - `set_reference_text(..., source=...)` now records `meta["reference_text_source"]` (e.g. `asr`, `manual`).

- `abstractvoice/vm/tts_mixin.py`
  - Default VoiceCloner uses `reference_text_whisper_model="small"` for better auto-fallback quality.

- Tests
  - Added `tests/test_cloning_reference_text_autofallback.py` (mocks STT to keep unit tests fast).

---

## Validation

- Tests: **34 passed, 2 skipped**
