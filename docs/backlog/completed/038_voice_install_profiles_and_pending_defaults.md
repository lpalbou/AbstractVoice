## Task 038: Voice install profiles and AbstractCore plugin defaults

**Date**: 2026-05-08  
**Status**: Completed  
**Priority**: P1  
**Promoted from**: `docs/backlog/proposed/2026-05-08_voice_install_profiles_and_pending_defaults.md`  
**Completed**: 2026-05-08  

**Superseded note**: Task 040 changed the release decision for `0.9.1`:
direct `VoiceManager()` and `auto` are now remote-first. The lightweight base
and plugin-default work from this task remains part of the release.

---

## Main goals

- Keep direct `VoiceManager()` library defaults local-first (`auto` resolves to Piper).
- Keep lightweight Core/Gateway plugin integrations remote-first by default.
- Make the AbstractCore capability plugin default to OpenAI remote TTS/STT with `OPENAI_API_KEY` fallback.
- Preserve owner config and `ABSTRACTVOICE_*` env overrides.
- Ensure OpenAI remote TTS lists its own hosted/custom voice profiles and model
  catalog instead of leaking Piper/local model display.
- Document install-profile boundaries, same-server recursion risk, and local web example boundaries.

## Secondary goals

- Keep local engines out of the base install.
- Avoid Gateway-specific env mutation in AbstractVoice.
- Keep the local/example web UI documented as local/dev, not as the production server boundary.

---

## Context / problem

Task 037 made the base package lightweight and moved Piper, faster-whisper, and
audio-device dependencies behind explicit extras. That left two valid default
contexts:

- Direct library use should remain local-first for existing `VoiceManager()` callers.
- AbstractCore/Gateway server integration should be remote-first so lightweight
  server images do not require local voice runtimes.

The pending plugin default diff had the right direction, but it needed cleanup
before it could be treated as accepted work.

---

## Current code reality

- `pyproject.toml` already keeps the base dependencies to `numpy`, `requests`,
  and `appdirs`.
- Remote OpenAI/OpenAI-compatible TTS/STT adapters already use the base HTTP
  dependency path.
- `VoiceManager(tts_engine="auto", stt_engine="auto")` remains local-first.
- The AbstractCore plugin constructs `VoiceManager` lazily and caches instances
  by configuration.
- The pending plugin diff defaulted Core integrations to OpenAI remote engines,
  but owner config booleans used `bool(value)`, so strings like `"false"` became
  truthy.

---

## Constraints

- Do not move production OpenAI-compatible HTTP serving into AbstractVoice.
- Do not add local voice engines to the base install.
- Apple/GPU profile aliases, when present, must map to real Voice-owned local dependencies or be
  documented compatibility aliases; they must not make the base install heavier.
- Do not write outside this repository folder.
- Treat unrelated pending changes as suspect and do not build on them.

---

## Decision

**Chosen approach**: accept the plugin remote-default direction after tightening
configuration parsing, tests, and docs.

**Why**:
- It matches the lightweight install profile from Task 037.
- It preserves direct `VoiceManager()` behavior for library users.
- It keeps AbstractVoice as the voice backend and AbstractCore/Gateway as the
  production routing/server layer.

---

## Implementation plan

- Harden plugin boolean coercion for env and owner config.
- Add tests for OpenAI defaults, env overrides, owner config overrides, string
  boolean handling, and missing `OPENAI_API_KEY` errors.
- Add tests for OpenAI remote TTS profile/model listing, including custom voice
  discovery when the provider exposes it.
- Add a live OpenAI regression that skips without a real `OPENAI_API_KEY` and
  confirms default profile/model listing through `VoiceManager`.
- Update user-facing docs for Core/Gateway remote defaults and recursion risk.
- Move this promoted proposed item to completed with validation results.

---

## Success criteria

- Plugin default path selects `tts_engine="openai"` and `stt_engine="openai"`.
- `OPENAI_API_KEY` is passed through as the remote API key.
- Owner config and env can select `openai-compatible` with a remote base URL.
- String config values such as `"false"` and `"0"` are not treated as truthy.
- Missing OpenAI credentials fail with a clear error.
- Docs explain that same-server compatible base URLs recurse.
- Docs explain that the local web UI is not the production Core/Gateway server
  surface.
- `VoiceManager(tts_engine="openai").get_profiles()` exposes OpenAI hosted
  voices and discovers provider/account voice profiles when available.
- `VoiceManager(tts_engine="openai").list_available_models()` reports OpenAI
  voices and discovered/configured TTS model ids.

---

## Test plan

- `pytest -q tests/test_abstractcore_plugin.py tests/test_remote_openai_compatible_adapters.py tests/test_lightweight_import_boundaries.py tests/test_dependency_check.py`
- `pytest -q`

---

## Report

### Summary

- Kept the AbstractCore plugin default remote-first for lightweight server
  integrations while leaving direct `VoiceManager()` local-first.
- Added shared boolean coercion so owner config/env strings like `"false"`,
  `"off"`, and `"0"` behave correctly.
- Added OpenAI TTS profile/model listing so `VoiceManager.get_profiles()` can
  discover provider/account voice profiles and `list_available_models()` reports
  remote voice/model entries instead of falling back to Piper catalog display.
- Added a skippable live OpenAI regression for the default `OPENAI_API_KEY`
  path; with a real key it verifies hosted voice profiles and TTS model
  discovery through `VoiceManager(tts_engine="openai")`.
- Added plugin tests for env overrides, owner config string booleans, and clear
  missing-key errors.
- Updated README/API/installation/getting-started docs so server examples show
  remote provider configuration and warn against recursive same-server
  `openai-compatible` base URLs.
- Clarified that the local FastAPI web UI is a local/dev example surface and
  does not inherit AbstractCore/Gateway auth or browser-origin policy.
- Added an Unreleased changelog entry. The package version was not bumped in
  this task because no release/tag was requested.

### Files touched

- `abstractvoice/integrations/abstractcore_plugin.py`
- `abstractvoice/adapters/tts_openai_compatible.py`
- `tests/test_abstractcore_plugin.py`
- `tests/test_remote_openai_compatible_adapters.py`
- `README.md`
- `docs/api.md`
- `docs/getting-started.md`
- `docs/installation.md`
- `docs/adr/0001-local_assistant_out_of_box.md`
- `docs/backlog/README.md`
- `CHANGELOG.md`
- `docs/backlog/completed/038_voice_install_profiles_and_pending_defaults.md`

### Validation

- Live OpenAI regression: `pytest -q tests/test_remote_openai_compatible_adapters.py::test_live_openai_default_lists_tts_models_and_voice_profiles -q` -> 1 passed.
- Focused: `pytest -q tests/test_abstractcore_plugin.py tests/test_remote_openai_compatible_adapters.py tests/test_lightweight_import_boundaries.py tests/test_dependency_check.py` -> 27 passed.
- Full suite: `pytest -q` -> 142 passed, 8 skipped, 2 warnings.
- Known warnings: `webrtcvad` imports deprecated `pkg_resources` in the full-mode echo-gate test path.

### Post-completion insights

- Most install-profile work was already completed by Task 037; this task was the
  smaller follow-up that made plugin defaults match that profile.
- The docs needed a stronger warning that compatible remote URLs must not point
  back at the same Core server route using the plugin fallback.
- Gateway/Core docs outside this repository were not updated because this task's
  write scope was limited to the AbstractVoice folder.

### Residual risks

- OpenAI's public voice/model-listing surface may vary by account and API
  version. AbstractVoice now falls back to the known hosted TTS model ids and
  built-in voice profiles when remote listing is unavailable.
