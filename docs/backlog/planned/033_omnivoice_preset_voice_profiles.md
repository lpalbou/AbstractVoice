## Task 033: OmniVoice preset voice profiles (male/female sets)

**Date**: 2026-04-06  
**Status**: Planned  
**Priority**: P2  

---

## Main goals

- Ship a small set of **curated OmniVoice “voice profiles”** that users can select easily:
  - **3 male** presets
  - **3 female** presets
- Make profiles **stable across turns** (same voice within a discussion).
  - For strong persistence, prefer anchoring the voice in a cached, tokenized prompt (`voice_clone_prompt`) rather than relying on voice-design RNG alone.

## Secondary goals

- Provide a clean cross-engine UX in the REPL:
  - `/profile list` (lists profiles for the currently selected `tts_engine`)
  - `/profile <name>` (applies the profile to the currently selected `tts_engine`)
  - `/profile show` (shows the active profile, if any)
- Provide a library surface (non-REPL) via a common engine interface:
  - `selected_engine.get_profiles()`
  - `selected_engine.set_profile(selected_profile_id)`

---

## Context / problem

OmniVoice supports “voice design” via:
- `instruct` (voice attributes like age/gender/pitch/accent)
- `seed` (stability of the designed voice across turns)
- sampling controls (`position_temperature`, `class_temperature`, etc.)

Today, users must assemble these settings manually. This is error-prone and makes it harder to quickly compare voices or provide a consistent “default voice pack” UX.
In practice, voice-design determinism is **best-effort** (accelerators like MPS can remain nondeterministic). For robust persistence, it is preferable to cache a tokenized reference prompt once and reuse it for subsequent synthesis.

---

## Constraints

- Avoid bundling large binary assets (no shipping WAVs).
- Keep offline-first semantics for local engines; OmniVoice is optional and heavy, so presets must not force downloads.
- Presets should be **documented as best-effort**:
  - accelerators (MPS/CUDA) can introduce nondeterminism
  - different OmniVoice versions may shift perceptual output slightly

---

## Proposed design

### 1) Dependency: shared “voice profile” abstraction (Task 036)

OmniVoice profiles should be implemented through a common cross-engine interface so that:
- the REPL can expose a single `/profile ...` command (no `/omnivoice profile ...` special-casing)
- AbstractCore integrations can forward profile selection without knowing per-engine parameters

See: `docs/backlog/planned/036_voice_profile_abstraction.md`.

Minimum viable profile payload for OmniVoice:

- `engine: "omnivoice"`
- `name: str`
- `params`:
  - `instruct: str`
  - `seed: int`
  - `num_step: int`
  - `guidance_scale: float`
  - `position_temperature: float`
  - `class_temperature: float`
  - optional persistence:
    - `persistent_prompt: bool` (enable prompt-token caching)
    - `prompt_text: str` (reference prompt text used to build the cached prompt)
    - `prompt_duration_s: float` (reference prompt duration)

### 2) Preset set: 6 curated profiles

Example naming convention:
- `female_01`, `female_02`, `female_03`
- `male_01`, `male_02`, `male_03`

Each profile should include:
- a short `instruct` string built from the documented “valid items”
- a fixed `seed` for stability
- a quality preset mapping (`low|standard|high`; aliases: `fast`, `balanced`) or explicit `num_step` defaults

### 3) Storage + override rules

Profiles can live as a simple JSON asset, e.g.:
- `abstractvoice/assets/voice_profiles/omnivoice_profiles.json`

Rules:
- Built-in profiles are read-only defaults.
- Users can provide custom profiles via a local config override path (optional).

---

## Implementation plan

- Add built-in profiles (JSON asset + loader) for OmniVoice.
- Implement OmniVoice adapter profile provider methods:
  - `get_profiles()` (returns the available OmniVoice profiles)
  - `set_profile(profile_id)` (applies `instruct`/`seed`/temps)
  - `get_active_profile()` (best-effort)
- Wire the REPL `/profile ...` command to the active engine via Task 036 (no engine-specific profile commands).
- Update docs (`docs/repl_guide.md`, `docs/faq.md`) with examples and determinism caveats.

---

## Success criteria

- A user can run:
  - `/tts_engine omnivoice`
  - `/profile female_01`
  - `/speak ...`
  - and get a stable voice across turns without needing to remember parameter details.
- Profiles are deterministic **best-effort** when `seed` is fixed; for strong persistence, profiles can build and reuse a cached tokenized prompt (one-time cost, then fast).

