## Task 039: AbstractCore plugin voice catalog surface

**Date**: 2026-05-08
**Status**: Completed
**Priority**: P1
**Promoted from**: `docs/backlog/proposed/2026-05-08_abstractcore_plugin_voice_catalog_surface.md`
**Completed**: 2026-05-08

**Superseded note**: Task 040 changed the default policy after this task:
direct `VoiceManager()` is now remote-first too. The catalog surface from this
task remains current.

---

## Main goals

- Expose voice profile and TTS model discovery through the AbstractCore capability plugin boundary.
- Keep voice-specific catalog semantics inside AbstractVoice instead of requiring Core/Gateway to call private `_get_vm()` internals.
- Preserve plugin registration-time lightness.

## Secondary goals

- Keep the plugin methods thin and JSON-safe.
- Document the new plugin catalog methods in the public integrator contract.
- Update `llms*.txt` so agents can find the catalog surface without reading the full codebase.

---

## Context / problem

Task 038 added OpenAI/OpenAI-compatible voice/profile/model discovery below the
plugin boundary:

- `OpenAICompatibleTTSAdapter.get_profiles()` exposes built-in, configured, and
  remote provider voice profiles.
- `OpenAICompatibleTTSAdapter.list_available_models()` returns a remote-shaped
  catalog containing available/configured TTS model ids per voice/profile.
- `VoiceManager.get_profiles(kind="tts")` and `VoiceManager.list_available_models()`
  delegate to the active adapter.

The AbstractCore capability plugin only exposed synthesis/transcription methods,
so Core/Gateway integrations would need private access to the cached
`VoiceManager` to publish dynamic voice catalogs.

---

## Constraints

- Do not add local engines to the base install.
- Do not move production OpenAI-compatible HTTP routing into AbstractVoice.
- Do not make catalog discovery run during plugin registration or shallow
  capability discovery.
- Do not mutate AbstractVoice environment variables per Core/Gateway request.
- Do not write outside the AbstractVoice repository folder.

---

## Decision

**Chosen approach**: implement a thin optional catalog surface on `_VoiceCapability`:

- `list_profiles(kind="tts") -> list[dict]`
- `list_tts_models() -> list[str]`
- `voice_catalog() -> dict`

**Why**:
- The adapter/manager already own voice catalog discovery and caching behavior.
- Core/Gateway can now ask the backend for JSON-safe catalog data without
  depending on private plugin internals.
- The new methods are invoked only when callers request catalog data, preserving
  lightweight plugin registration.

---

## Implementation plan

- Add JSON-safe `VoiceProfile` serialization helpers to
  `abstractvoice/integrations/abstractcore_plugin.py`.
- Add TTS model id extraction from nested/remote-shaped catalogs.
- Add `_VoiceCapability` discovery methods that delegate to the active
  `VoiceManager`.
- Add focused unit coverage with an injected fake `VoiceManager`.
- Update README, API docs, changelog, and `llms*.txt`.
- Move this promoted proposed item to completed with validation results.

---

## Success criteria

- Core/Gateway can call the plugin voice backend to list profiles and TTS model ids.
- Returned profiles/catalog data is JSON-safe.
- The combined `voice_catalog()` response includes active profile/model when available.
- Existing TTS/STT plugin behavior is unchanged.
- Tests pass for the new catalog surface and existing remote/plugin behavior.

---

## Test plan

- `pytest -q tests/test_abstractcore_plugin.py`
- `pytest -q tests/test_abstractcore_plugin.py tests/test_remote_openai_compatible_adapters.py tests/test_lightweight_import_boundaries.py tests/test_dependency_check.py`
- `pytest -q`
- `git diff --check`

---

## Report

### Summary

- Added `_VoiceCapability.list_profiles(...)`, `_VoiceCapability.list_tts_models()`,
  and `_VoiceCapability.voice_catalog()`.
- Added JSON-safe profile serialization and deduplicated TTS model id extraction
  from nested voice catalogs.
- Added focused plugin tests using an injected fake `VoiceManager`.
- Updated `README.md` and `docs/api.md` to document the plugin catalog surface.
- Updated `llms.txt` and `llms-full.txt` in the existing spec-aligned style so
  agents see the new plugin discovery boundary.
- Added an Unreleased changelog entry. The package version was not bumped because
  no release/tag was requested.

### Files touched

- `abstractvoice/integrations/abstractcore_plugin.py`
- `tests/test_abstractcore_plugin.py`
- `README.md`
- `docs/api.md`
- `llms.txt`
- `llms-full.txt`
- `CHANGELOG.md`
- `docs/backlog/completed/039_abstractcore_plugin_voice_catalog_surface.md`

### Validation

- Focused plugin: `pytest -q tests/test_abstractcore_plugin.py` -> 10 passed.
- Focused remote/plugin set: `pytest -q tests/test_abstractcore_plugin.py tests/test_remote_openai_compatible_adapters.py tests/test_lightweight_import_boundaries.py tests/test_dependency_check.py` -> 28 passed.
- Full suite: `pytest -q` -> 143 passed, 8 skipped, 2 warnings.
- Diff check: `git diff --check` -> passed.

### Post-completion insights

- This backlog item was worth doing because Task 038 created the underlying
  catalog data, but Core/Gateway still needed a public backend boundary for it.
- Context7 MCP did not expose resources in this session; `llms*.txt` updates
  followed the repo's completed `llms.txt` best-practices task and current
  spec-shaped manifests.

### Residual risks

- AbstractCore still needs to decide which HTTP route or capability API shape
  should surface these methods to external clients. AbstractVoice now provides
  the backend data contract, but Core owns routing, auth, and browser policy.
