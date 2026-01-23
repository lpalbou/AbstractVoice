## Task 017: Remove legacy Coqui model management (no legacy scaffolding)

**Date**: 2026-01-23  
**Status**: Completed  
**Priority**: P0  

---

## Summary

Removed Coqui-specific “model management” and legacy TTS pathways from AbstractVoice core to keep the architecture **clean, Piper-first**, and avoid conceptual drift.

This preserves the integrator-facing `VoiceManager` contract (speak/listen/transcribe/pause/resume/stop) while eliminating legacy surfaces that encouraged duplicated catalogs and doc/test drift.

---

## Changes

- **Deleted legacy modules**
  - Removed `abstractvoice/coqui_model_manager.py`, `abstractvoice/simple_model_manager.py`, and `abstractvoice/instant_setup.py`.
  - Removed `abstractvoice/vm/mm_mixin.py` and the corresponding inheritance from `VoiceManager`.

- **Piper-only core behavior**
  - `VoiceManager` now supports only `tts_engine in {"auto","piper"}` (errors otherwise).
  - `set_language()` and `set_voice()` are Piper-backed; `voice_id` remains best-effort.

- **Replaced legacy TTSEngine**
  - Rewrote `abstractvoice/tts/tts_engine.py` to contain only reusable audio utilities:
    - `NonBlockingAudioPlayer`
    - `apply_speed_without_pitch_change` (best-effort, optional librosa)
  - Updated `abstractvoice/tts/__init__.py` exports accordingly.

- **CLI + tests aligned**
  - Removed the `download-models` command path from `abstractvoice/examples/voice_cli.py`.
  - Updated CLI voice listing to use the Piper-backed `list_available_models()`.
  - Updated `tests/test_fresh_install.py` to assert no legacy model-management API leaks into `abstractvoice`.

- **Dependencies**
  - Removed `coqui-tts` and PyTorch stack from optional dependency groups in `pyproject.toml`.
  - Added `audio-fx` extra for optional `librosa` time-stretching.

- **Docs**
  - Rewrote `docs/model-management.md` to reflect Piper-only core.
  - Updated `docs/architecture.md`, `docs/installation.md`, `docs/multilingual.md` to remove Coqui/XTTS-specific guidance from the main docs.

---

## Validation

- Tests: **29 passed, 1 skipped**

