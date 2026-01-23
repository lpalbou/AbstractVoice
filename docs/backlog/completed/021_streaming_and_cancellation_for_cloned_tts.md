## Task 021: Streaming + cancellation for cloned TTS (REPL responsiveness)

**Date**: 2026-01-23  
**Status**: Completed  
**Priority**: P0  

---

## Summary

Implemented progressive playback for cloned voices and best-effort cancellation:

- cloned TTS now enqueues chunks progressively into the existing audio player
- `stop_speaking()` cancels ongoing cloned synthesis and stops playback immediately

This improves perceived latency for longer answers and makes REPL interruption behave as expected.

---

## Changes

- `abstractvoice/tts/adapter_tts_engine.py`
  - Added `begin_playback()` and `enqueue_audio()` to support streaming playback without re-triggering playback-start callbacks per chunk.

- `abstractvoice/cloning/engine_f5.py`
  - Added `infer_to_audio_chunks(...)` generator (uses `f5_tts.infer.utils_infer.infer_batch_process(..., streaming=True)`).

- `abstractvoice/cloning/manager.py`
  - Added `speak_to_audio_chunks(...)` to expose chunk generation per voice id.

- `abstractvoice/vm/tts_mixin.py`
  - Cloned `VoiceManager.speak(..., voice=...)` now runs synthesis in a background thread and progressively enqueues audio chunks.
  - Added best-effort cancellation token; `stop_speaking()` sets it.

- Tests
  - Added `tests/test_cloned_tts_cancellation.py`

---

## Validation

- Tests: **33 passed, 2 skipped**

