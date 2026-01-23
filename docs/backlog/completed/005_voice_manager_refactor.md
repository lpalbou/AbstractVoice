## Task 005: Refactor `voice_manager.py` into small focused modules

**Status**: Completed  
**Priority**: P0  

---

## Summary

`VoiceManager` was refactored into a thin public façade plus small focused internal modules to reduce cognitive load, prevent accidental coupling, and keep responsibilities clear.

---

## What changed

- `abstractvoice/voice_manager.py` is now a **thin façade** that re-exports `VoiceManager`.
- New internal package `abstractvoice/vm/` contains the implementation split by responsibility:
  - `manager.py`: init/wiring and engine selection
  - `tts_mixin.py`: TTS orchestration (speak/pause/stop, language/voice selection, headless bytes/file synthesis)
  - `stt_mixin.py`: STT + listen/pause/resume + modes
  - `core.py`: lifecycle callback wiring + cleanup
  - `mm_mixin.py`: MM (model management) façade methods
- New `abstractvoice/config/voice_catalog.py` centralizes language and voice catalog constants.

---

## Why this is better

- The public API stays stable, but the implementation becomes navigable and modular.
- Internal boundaries are clearer (TTS vs STT vs lifecycle vs catalogs), enabling targeted changes without ripple effects.

---

## Validation

- Full test suite: **26 passed, 1 skipped**

