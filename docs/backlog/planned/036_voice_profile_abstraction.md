## Task 036: Voice profile abstraction (cross-engine) + API exposure

**Date**: 2026-04-07  
**Status**: Implemented  
**Priority**: P1  

---

## Main goals

- Define a **single cross-engine “voice profile” abstraction** that can represent:
  - local preset voices (OmniVoice / AudioDiT / Piper curated presets)
  - commercial provider voices (e.g. OpenAI built-ins and custom voice ids)
  - (optionally later) cloned voices stored in the local voice store
- Expose a **common interface** on engines so callers can:
  - list available profiles for the active engine
  - select/apply a profile by id
  - query the currently active profile
- Provide a **single REPL UX**:
  - `/profile list`
  - `/profile <name>`
  - `/profile show`
- Ensure the abstraction is usable for:
  - **third-party integrations** (library API)
  - **AbstractCore integrations** (capability plugin + tool wiring)

---

## Context / problem

We currently plan per-engine profile UX (e.g. `/omnivoice profile ...`, `/audiodit profile ...`), which causes:
- duplicated REPL command surfaces per engine
- duplicated integration glue in AbstractCore/server layers
- unclear “source of truth” for what a “profile” means across engines

Instead, profiles should be engine-agnostic at the UX/API layer:
- users select an engine (`tts_engine`)
- then apply a profile to the **currently selected engine**

At the engine layer this becomes:
- `selected_engine.get_profiles()`
- `selected_engine.set_profile(selected_profile_id)`

---

## Proposed design

### 1) Types

Introduce a provider-agnostic profile type (exact module path is flexible):

- `VoiceProfile`
  - `engine_id: str` (e.g. `omnivoice|audiodit|piper|openai|...`)
  - `profile_id: str` (stable id within the engine, e.g. `female_01`, `en_female_01`, `alloy`)
  - `label: str` (human-friendly)
  - `description: str | None`
  - `params: dict[str, Any]` (engine-specific; validated best-effort by the engine)
  - `tags: dict[str, str] | None` (e.g. gender, language, style)
  - `provenance: dict[str, Any] | None` (optional; required for consent-gated voices)

Notes:
- Keep `profile_id` **engine-local** so `/profile female_01` works naturally once an engine is selected.
- For cross-engine addressing (optional), we can support a qualified id convention like `omnivoice:female_01`.

### 2) Common engine interface

Add a shared profile interface that engines can implement (e.g. Protocol / base mixin):

- `get_profiles() -> list[VoiceProfile]`
- `set_profile(profile_id: str) -> bool`
- `get_active_profile() -> VoiceProfile | None`

Design constraints:
- Default implementations should exist (return empty list / False / None) so engines without profiles remain compatible.
- Profile application must be **idempotent** and safe to call multiple times.

### 3) VoiceManager surface (library API)

Expose an engine-agnostic surface on `VoiceManager`:

- `vm.get_profiles(kind="tts") -> list[VoiceProfile]` (delegates to the active TTS engine)
- `vm.set_profile(profile_id: str, *, kind="tts") -> bool`
- `vm.get_active_profile(kind="tts") -> VoiceProfile | None`

Optional: allow a per-call override in `speak_to_bytes`/`speak_to_file` (only if it remains non-ambiguous and thread-safe).

### 4) REPL UX

Implement a single `/profile` command:
- `/profile list` → lists profiles for the active `tts_engine`
- `/profile <name>` → applies the profile for the active `tts_engine`
- `/profile show` → prints current profile + engine id

No engine-specific commands like `/omnivoice profile ...` or `/audiodit profile ...`.

### 5) AbstractCore integration exposure

We want profiles to be forwardable through AbstractCore/server layers without engine-specific glue.

Proposed capability/plugin behavior (exact contract to align with AbstractCore):
- Accept an optional `profile` parameter on TTS calls (per request).
- Apply `profile` to the active engine **before synthesis**.
- Expose the effective `engine_id` + `profile_id` in returned metadata/artifact metadata when available.

Thread-safety note:
- If a single `VoiceManager` is shared across concurrent requests, “setting a profile” mutates engine state.
- For production servers, the preferred pattern may be:
  - per-session `VoiceManager` instances, or
  - per-request isolated engines, or
  - a lock + “apply profile → synth → restore previous profile” pattern (if supported).

This task should define the recommended pattern and document it.

---

## Related planned work

- OmniVoice presets: `docs/backlog/planned/033_omnivoice_preset_voice_profiles.md`
- AudioDiT presets: `docs/backlog/planned/034_audiodit_preset_voice_profiles.md`
- Piper true voice selection + presets: `docs/backlog/planned/035_piper_voice_profiles_and_voice_selection.md`
- OpenAI voices mapping: `docs/backlog/planned/013_openai_compatible_audio_endpoint.md`

---

## Implementation plan

- Add `VoiceProfile` type + engine interface (default no-op implementations).
- Implement `/profile` in the REPL (wired to `VoiceManager`).
- Implement profile support in at least one engine first (OmniVoice or Piper) to validate the contract end-to-end.
- Add unit tests:
  - listing profiles
  - applying profiles
  - ensuring engines without profiles behave unchanged
- Document:
  - determinism caveats (seeds, accelerators)
  - concurrency + server guidance

---

## Success criteria

- A user can run:
  - `/tts_engine omnivoice`
  - `/profile list`
  - `/profile female_01`
  - `/speak ...`
- Third-party integration can call:
  - `vm.get_profiles(...)` and `vm.set_profile(...)`
- AbstractCore integration can forward a `profile` parameter without engine-specific code.

---

## Implementation notes (2026-04-07)

Implemented in this repo:
- Type + loader: `abstractvoice/voice_profiles.py`
- Adapter interface: `abstractvoice/adapters/base.py` (optional `get_profiles/set_profile/get_active_profile`)
- VoiceManager API: `abstractvoice/vm/manager.py`
- REPL UX: `abstractvoice/examples/cli_repl.py` (`/profile list|show|<id>`)
- Built-in demo pack (OmniVoice): `abstractvoice/assets/voice_profiles/omnivoice_profiles.json`
- AbstractCore tool wiring: `abstractvoice/integrations/abstractcore.py` (`voice_profile_list`, `voice_profile_set`)

Validation on OmniVoice:
- Prefetch OmniVoice weights: `python -m abstractvoice download --omnivoice`
- In REPL: `/tts_engine omnivoice` → `/profile list` → `/profile female_01` → `/speak ...`
