## Task 035: Piper voice profiles (true voice selection + presets)

**Date**: 2026-04-06  
**Status**: Planned  
**Priority**: P2  

---

## Main goals

- Make Piper support **true voice selection** (not just language selection).
- Add a small set of **curated Piper “voice profiles”** (e.g. 3 male + 3 female for `en`, and a smaller curated set for `fr` where available).

## Secondary goals

- Keep offline-first semantics:
  - listing voices is offline
  - downloads are explicit / opt-in (REPL already uses `allow_downloads=False`)
- Keep the API stable for existing users relying on “one default voice per language”.

---

## Context / problem

Today, `PiperTTSAdapter.PIPER_MODELS` is a **single voice per language** mapping, and:
- `VoiceManager.set_voice(language, voice_id)` is explicitly “best-effort metadata”
- the REPL `/setvoice` command suggests voice selection, but it can’t actually change the underlying Piper voice

This blocks:
- Piper preset profiles (male/female sets)
- meaningful voice comparisons in benchmarks
- better UX for users who want a different default voice without changing engines

---

## Constraints

- Do not introduce system dependencies (Piper is a core “zero-deps” path).
- Avoid pulling Hugging Face Hub libraries into the critical path; current downloads use direct HTTPS for robustness.
- Keep `docs/model-management.md` / `docs/installation.md` aligned with any new download behavior.

---

## Proposed design

### Dependency: shared “voice profile” abstraction (Task 036)

Curated Piper presets should be exposed through the same cross-engine profile interface as other engines so:
- the REPL can use a single `/profile ...` command regardless of engine
- AbstractCore / third-party integrations can forward profile selection without Piper-specific logic

See: `docs/backlog/planned/036_voice_profile_abstraction.md`.

### 1) Expand Piper model catalog to support multiple voices per language

Change `PIPER_MODELS` shape from:

- `lang -> (hf_path, model_filename)`

to:

- `lang -> { voice_id -> {hf_path, model_filename, quality, size_mb, description} }`

This allows:
- listing multiple voices per language
- selecting a voice by id

### 2) Make `set_voice(language, voice_id)` actually switch the model

- Add `PiperTTSAdapter.set_voice(language, voice_id)` (or extend `set_language` signature) so it:
  - resolves the correct HF path/model filename
  - ensures the specific ONNX + JSON are downloaded/cached (explicitly)
  - loads that voice into `_voice`
  - updates `get_info()` / `list_available_models()` accordingly

`VoiceManager.set_voice(...)` should delegate to the adapter when active engine is Piper.

### 3) Preset profiles (optional on top)

Once multi-voice selection works, define a small curated profile list:
- `piper_en_female_01`, `piper_en_male_01`, etc.
- Each profile is essentially `{engine:"piper", language:"en", voice_id:"..."}`.

These curated presets should be surfaced via the shared profile interface:
- `get_profiles()` returns the curated list (and optionally the full catalog as “un-curated” profiles)
- `set_profile(profile_id)` applies `{language, voice_id}` to the adapter

---

## Implementation plan

- Update `PiperTTSAdapter`:
  - catalog structure
  - `list_available_models()` returns all voices
  - implement true voice switching
- Update `VoiceManager.set_voice(...)` to actually switch voices (Piper only).
- Wire curated presets into the shared profile system (Task 036) so users can use `/profile ...`.
- Update REPL `/setvoice` help text if needed (as the “low-level” selector; profiles are the “curated” selector).
- Add tests:
  - catalog listing shape
  - set_voice selects the right model file paths (mock filesystem; no network)

---

## Success criteria

- `/setvoice en.<voice_id>` changes the actual Piper voice used for synthesis.
- With `tts_engine=piper`, users can select curated presets via `/profile ...` (Task 036) without needing to remember raw `voice_id` values.
- Users can select from a small curated preset list without editing code.
- Existing behavior remains intact when users never call `set_voice` (default per-language voice still works).

