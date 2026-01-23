## Task 018: REPL voice cloning workflow + real integration test

**Date**: 2026-01-23  
**Status**: Completed  
**Priority**: P0  

---

## Summary

Closed two gaps:

1) **Tests**: added a gated integration test that actually synthesizes from the HAL9000 reference WAV when cloning deps are available.  
2) **REPL**: added seamless commands to clone, list, select, and speak using cloned voices, with a default seeded `hal9000` entry.

---

## Changes

- **REPL (`abstractvoice/examples/cli_repl.py`)**
  - Seed `hal9000` in the local voice store on startup if `audio_samples/hal9000/` exists.
  - New commands:
    - `/clones` (list cloned voices)
    - `/clone <path> [name]`
    - `/clone-my-voice` (records a short prompt to WAV and clones it)
    - `/tts_voice piper` or `/tts_voice clone <id-or-name>`
  - All speaking now respects the selected cloned voice via `VoiceManager.speak(..., voice=...)`.

- **VoiceManager playback**
  - Added `voice: str | None` to `speak()` to allow local playback of cloned voices (decode WAV → mono → resample → play).
  - Added `AdapterTTSEngine.play_audio_array()` to reuse lifecycle callbacks + low-latency playback.

- **Audio utilities**
  - Added `abstractvoice/audio/` with:
    - `linear_resample_mono()` (lightweight resampling)
    - `record_wav()` (for `/clone-my-voice`)

- **Tests**
  - Added `tests/test_voice_cloning_integration_hal9000.py` (skipped by default).
    - Enable with: `ABSTRACTVOICE_RUN_CLONING_TESTS=1`
    - Also requires `f5-tts_infer-cli` to be available (`abstractvoice[cloning]` installed).

- **Docs**
  - Updated `docs/repl_guide.md` and `docs/public_api.md` to reflect the optional `voice` parameter and the cloning workflow.

---

## Validation

- Tests: **30 passed, 2 skipped**

