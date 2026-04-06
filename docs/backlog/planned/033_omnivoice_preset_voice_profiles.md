## Task 033: OmniVoice preset voice profiles (male/female sets)

**Date**: 2026-04-06  
**Status**: Planned  
**Priority**: P2  

---

## Main goals

- Ship a small set of **curated OmniVoice “voice profiles”** that users can select easily:
  - **3 male** presets
  - **3 female** presets
- Make profiles **stable across turns** (same voice within a discussion) and **portable** across machines when the same settings are used.

## Secondary goals

- Provide a clean UX in the REPL:
  - `/omnivoice profile list`
  - `/omnivoice profile <name>`
- Provide a library surface (non-REPL) that can select a profile by id/name.

---

## Context / problem

OmniVoice supports “voice design” via:
- `instruct` (voice attributes like age/gender/pitch/accent)
- `seed` (stability of the designed voice across turns)
- sampling controls (`position_temperature`, `class_temperature`, etc.)

Today, users must assemble these settings manually. This is error-prone and makes it harder to quickly compare voices or provide a consistent “default voice pack” UX.

---

## Constraints

- Avoid bundling large binary assets (no shipping WAVs).
- Keep offline-first semantics for local engines; OmniVoice is optional and heavy, so presets must not force downloads.
- Presets should be **documented as best-effort**:
  - accelerators (MPS/CUDA) can introduce nondeterminism
  - different OmniVoice versions may shift perceptual output slightly

---

## Proposed design

### 1) Introduce a shared “voice profile” concept (recommended dependency)

This task is easiest if we have a cross-engine profile abstraction (see Task 013 improvements / “voice profile” concept).

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

### 2) Preset set: 6 curated profiles

Example naming convention:
- `female_01`, `female_02`, `female_03`
- `male_01`, `male_02`, `male_03`

Each profile should include:
- a short `instruct` string built from the documented “valid items”
- a fixed `seed` for stability
- a quality preset mapping (`fast|balanced|high`) or explicit `num_step` defaults

### 3) Storage + override rules

Profiles can live as a simple JSON asset, e.g.:
- `abstractvoice/assets/voice_profiles/omnivoice_profiles.json`

Rules:
- Built-in profiles are read-only defaults.
- Users can provide custom profiles via a local config override path (optional).

---

## Implementation plan

- Add built-in profiles (JSON asset + loader).
- Add `VoiceManager`/adapter helper to apply profile params (best-effort).
- Extend REPL command surface:
  - list profiles
  - select by name
  - show current profile
- Update docs (`docs/repl_guide.md`, `docs/faq.md`) with examples and determinism caveats.

---

## Success criteria

- A user can run:
  - `/tts_engine omnivoice`
  - `/omnivoice profile female_01`
  - `/speak ...`
  - and get a stable designed voice across turns without needing to remember parameter details.
- Profiles are deterministic “best-effort” when `seed` is fixed, and the docs clearly explain remaining nondeterminism risks.

