# Task 056: Normalize AbstractCore capability residency truth

**Date**: 2026-05-21
**Status**: Completed
**Priority**: P1
**Promoted from**: `docs/backlog/proposed/0056_normalize_abstractcore_capability_residency_truth.md`
**Completed**: 2026-05-21

---

## Main goals

- Make cloned TTS residency load responses distinguish first warm from reuse.
- Add `loaded` plus `engine_cached_before/after` for event truth, while keeping `resident` for
  backward compatibility (then deprecate `resident` once Core stops reading it).

## Context / problem statement

AbstractVoice already exposes the right AbstractCore-facing capability boundary:

- `available_providers()`
- `list_models(kind="tts"|"stt"|"cloning", provider=...)`
- `load_resident_model(...)`
- `list_resident_models(...)`
- `unload_resident_model(...)`

The residency implementation is intentionally clone-TTS focused. That is still the right scope:
base TTS and STT warmup should not be reported as loaded unless there is measured evidence and a
real in-process runtime to control.

The remaining issue is event truth. Core can list and unload loaded cloned TTS engines, but the
load response does not reliably tell Core whether the call created/warmed a new runtime or reused an
already cached one.

## Current code reality

- `abstractvoice/integrations/abstractcore_plugin.py`
  - `_VoiceCapability.load_resident_model(...)` implements real cloned-TTS warmup for
    `task="tts", provider="cloned"`.
  - `_VoiceCapability.list_resident_models(...)` reports process-local clone-engine state.
  - `_VoiceCapability.unload_resident_model(...)` unloads cloned TTS engines.
  - `_AudioCapability` returns structured `not_implemented_yet` for STT residency and lists no
    loaded STT runtimes.
- `VoiceManager.preload_cloning_engine(...)` delegates to clone-engine preload state.
- `abstractvoice/cloning/manager.py`
  - `VoiceCloner.preload_engine(...)` currently reports `engine_cached` based on the post-load
    `self._engines` mapping, but `_get_engine(...)` always inserts the engine into the cache.
    That makes `engine_cached` true even on a first warm, so Core cannot distinguish first load vs reuse.
- Core's `/acore/models/load` derives `loaded_new` from returned runtime fields. If AbstractVoice
  only reports post-load `engine_cached`, Core can misread a first warm as a reuse.

## Dependencies

- ADR: `docs/adr/0006_voice_overlays_are_package_owned_not_generic_adapters.md`
- Backlog:
  - `docs/backlog/completed/042_capability_residency_hooks.md`

## Decision

Keep the residency scope narrow. Do not add generic base TTS/STT residency now.

Fix the cloned-TTS response contract so AbstractCore can distinguish:

- engine was already cached before the request;
- engine became cached because of the request;
- the load call should be treated as "newly loaded" vs "already loaded" (if AbstractCore continues
  to expose `loaded_new`).

If the existing preload path already knows that a separate warmup step happened, it may also return
`warmed_new`, but that is not required for the first fix.

## Constraints

- Do not report configured or catalog-listed voice providers as loaded.
- Do not add fake base TTS/STT warmup just for symmetry with image or text.
- Do not leak prompts, voice-private data, API keys, or generated audio into residency records.
- Preserve existing `resident` and `engine_cached` fields for backward compatibility while adding
  better event-truth fields.

## Implementation plan

1. In the clone engine preload path, compute `engine_cached_before` before calling the engine
   factory/cache.
2. Return `engine_cached_before` and `engine_cached_after` in addition to the existing `engine_cached`.
3. Add `loaded` as a forward-looking alias for `resident` (do not remove `resident` until AbstractCore
   is updated to prefer `loaded` and tests confirm no regressions).
4. Propagate those fields through `VoiceManager.preload_cloning_engine(...)`.
5. Propagate those fields through `_VoiceCapability.load_resident_model(...)`.
6. Keep non-cloned TTS and STT residency returning explicit `not_implemented_yet`.

## Success criteria

- First cloned TTS warm returns `loaded=true`.
- Rewarming the same cloned TTS engine returns `loaded=true`.
- Core `/acore/models/loaded?task=tts` reports only real in-process clone-engine state.
- Base TTS and STT residency do not report false positives.

## Test plan

- Add plugin tests for first warm versus repeated warm.
- Keep existing non-cloned TTS/STT deferred-support tests.
- Add or run a Core integration test that exercises `/acore/models/load` for cloned TTS through the
  capability plugin.

---

## Report

### Summary

- Fixed cloned-TTS residency event truth by reporting whether the clone engine cache existed before
  the request (`engine_cached_before`) and after (`engine_cached_after`).
- Added `loaded` as a forward-looking alias for `resident` in residency responses (kept `resident`
  for backward compatibility).
- Updated plugin residency tests to cover first warm vs repeated warm and to assert `loaded` is
  present/consistent.
- Added a focused unit test for `VoiceCloner.preload_engine(...)` to lock down the new before/after
  cache-truth fields.
- Added `pythonpath = [\".\"]` to pytest config so `pytest` (entrypoint script) runs tests against the
  local checkout consistently (matching `python -m pytest` behavior).

### Files touched

- `abstractvoice/cloning/manager.py`
- `abstractvoice/integrations/abstractcore_plugin.py`
- `tests/test_abstractcore_plugin.py`
- `tests/test_voice_cloner_engine_dispatch.py`
- `pyproject.toml`
- `docs/backlog/completed/0056_normalize_abstractcore_capability_residency_truth.md`

### Validation

- Focused plugin residency: `pytest -q tests/test_abstractcore_plugin.py -k residency` -> passed.
- Focused cloner cache truth: `pytest -q tests/test_voice_cloner_engine_dispatch.py -k preload_reports` -> passed.
- CI/release gate: `python -m pytest -q -m "not integration and not model_download"` -> passed (203 passed, 3 skipped).
