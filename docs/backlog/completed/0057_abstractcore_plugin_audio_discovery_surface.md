# Task 057: AbstractCore Plugin Audio Discovery Surface

**Date**: 2026-05-21  
**Status**: Completed  
**Priority**: P2  
**Promoted from**: `docs/backlog/proposed/0057_abstractcore_plugin_audio_discovery_surface.md`  
**Completed**: 2026-05-21  

---

## Main goals

- Make AbstractVoice's **audio/STT** capability backend satisfy AbstractCore's generic discovery contract:
  - `available_providers(task=None)`
  - `list_models(task=None, provider=...|provider_id=...)`

## Context / problem

AbstractCore now has a small generic capability plugin discovery surface (`llm.capabilities.*` and
`/v1/capabilities/*`). In current AbstractVoice (including `0.10.12`), the voice backend
(`abstractvoice:default`) implements discovery, but the audio/STT backend (`abstractvoice:stt`)
does not.

Impact:

- `llm.audio.transcribe(...)` works (typed facade).
- But `llm.capabilities.available_providers("audio")` and `llm.capabilities.list_models("audio")`
  fail because the selected audio backend has no discovery methods.

This is not a residency/warmup task. It is only about consistent discovery across capability
backends so UIs/runtimes can ask one generic question.

## Constraints

- Discovery methods must remain import-light and must not load models.
- Keep `VoiceManager` public behavior stable.
- Do not claim model residency or warm state for STT/TTS; returning "unsupported" for load/unload is fine.

## Decision

Add minimal discovery methods to `_AudioCapability` inside
`abstractvoice/integrations/abstractcore_plugin.py`, delegating to existing provider/model listing
helpers already used by `_VoiceCapability`.

## Implementation plan

- Add `_AudioCapability.available_providers(task=None)`:
  - return the STT provider ids (and optionally known/active ids) in a JSON-safe payload.
- Add `_AudioCapability.list_models(task=None, provider=None, provider_id=None, kind=None)`:
  - accept either `provider` or `provider_id` for compatibility with AbstractCore's structural call filtering;
  - delegate to existing STT model listing (`list_stt_models(...)`) and provider/model resolution helpers.
- (Optional) Add `_AudioCapability.list_operations(...)` returning `[]` (or a tiny record declaring STT) if it is trivial.

## Success criteria

- In AbstractCore with AbstractVoice installed:
  - `llm.capabilities.available_providers("audio")` returns a non-empty provider list when STT engines are available.
  - `llm.capabilities.list_models("audio")` returns model ids without loading a model.

## Test plan

- Add a lightweight AbstractVoice plugin unit test that instantiates `_AudioCapability` with a stub `owner`
  and asserts `available_providers` / `list_models` return JSON-safe values.
- Add/adjust an AbstractCore contract test only if needed (prefer plugin-owned tests here).

---

## Report

### Summary

- Added the missing AbstractCore generic discovery methods on the audio/STT backend (`abstractvoice:stt`):
  `_AudioCapability.available_providers(task=None)` and
  `_AudioCapability.list_models(task=None, provider=...|provider_id=...)`.
- Kept discovery import-light and model-load-free by delegating to existing config/env-based and static
  model catalogs already used by the voice backend.
- Added a focused unit test exercising the new discovery surface (including task filtering and
  `provider_id` compatibility).

### Files touched

- `abstractvoice/integrations/abstractcore_plugin.py`
- `tests/test_abstractcore_plugin.py`
- `docs/backlog/completed/0057_abstractcore_plugin_audio_discovery_surface.md`

### Validation

- `python -m pytest -q tests/test_abstractcore_plugin.py` -> passed (38 passed).
