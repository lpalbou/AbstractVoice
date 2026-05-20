## Task 044: Package-owned directed speech request and capability contract

**Date**: 2026-05-20  
**Status**: Proposed  
**Priority**: P1

---

## Main goals

- Define a package-owned request shape for advanced speech generation that is
  richer than the current `text + voice + speed + instructions` surface.
- Keep engine-specific behavior honest by requiring capability reporting and
  explicit degradation instead of silent best-effort guessing.
- Create the abstraction that Scenema-class and DramaBox-class engines would
  need before any engine-specific runtime work is promoted.

## Secondary goals

- Preserve the current `VoiceManager` public API while creating a cleaner
  internal contract for future engines.
- Give AbstractCore and other integrations a stable dict/JSON-facing surface
  they can forward without redefining package semantics.

---

## ADR status

- **Governing ADRs**:
  - `docs/adr/0005_torch_device_and_dtype_policy.md`
  - `docs/adr/0006_voice_overlays_are_package_owned_not_generic_adapters.md`
- **ADR impact**: Needs a new ADR or a revision that explicitly governs a
  package-owned `SpeechRequest` / capability degradation contract.

---

## Context / problem statement

AbstractVoice currently exposes a speech API that is still fundamentally
text-first:

- `TTSAdapter.synthesize(text)` in `abstractvoice/adapters/base.py`
- `VoiceManager.speak_to_bytes(text, voice, format)` in
  `abstractvoice/vm/tts_mixin.py`
- limited plugin controls such as `speed`, `quality_preset`, `instructions`,
  `profile`, and cloned `voice` in
  `abstractvoice/integrations/abstractcore_plugin.py`

That works for today's engines, but it is too narrow for engines whose native
surface includes:

- scene context;
- action cues;
- ambient/background audio;
- target pacing or target duration;
- output layout choices such as mono versus stereo;
- quality validation and regeneration policy.

If each new engine receives ad hoc kwargs, the package-level contract becomes
opaque and future integrations drift into engine-specific glue.

---

## Current code reality

- `abstractvoice/adapters/base.py` defines a thin TTS adapter contract centered
  on plain text synthesis plus optional profiles and quality presets.
- `abstractvoice/vm/tts_mixin.py` still routes per-request behavior mainly
  through mutable `VoiceManager` state such as `speed` and adapter profile
  selection.
- `abstractvoice/examples/web_ui.py` already contains a local `SpeechRequest`
  model, but it only covers:
  - text/input
  - voice
  - format
  - language
  - speed
  - sanitize flag
- `docs/adr/0006_voice_overlays_are_package_owned_not_generic_adapters.md`
  already establishes that package-owned selectors such as `provider`, `model`,
  `profile`, `voice`, `instructions`, and `quality_preset` must not be
  flattened into a fake generic adapter field.
- No first-class package-owned request object exists today for scene-aware or
  multi-step speech generation.

---

## Constraints

- Keep the existing public `VoiceManager` API stable.
- Do not overload `profile`, `voice`, or `model` with new hidden meanings.
- Preserve honest per-engine behavior reporting.
- Allow integrations to use dict/JSON payloads without importing internal
  classes.
- Keep simple engines simple; a richer contract must not force Piper-class
  backends to fake support they do not have.

---

## Research, options, and references

### Option A: keep the current text-first surface and add more per-engine kwargs

- **Pros**:
  - lowest immediate code churn;
  - no new request type to teach.
- **Cons**:
  - pushes semantics into undocumented kwargs;
  - makes capability reporting nearly impossible;
  - causes every integration layer to learn engine-specific exceptions;
  - does not scale to scene-aware engines.

### Option B: add a package-owned directed speech request and capability contract

- **Pros**:
  - keeps semantics owned by AbstractVoice;
  - lets engines declare `native`, `emulated`, or `unsupported` behavior;
  - keeps plugin and server layers thin and honest;
  - provides a clean target for Scenema-class engines.
- **Cons**:
  - requires contract design work before engine work;
  - likely needs a new ADR.

### Option C: let AbstractCore or another upstream integration define the richer request shape

- **Pros**:
  - pushes planning work out of this repo.
- **Cons**:
  - reverses the intended dependency direction;
  - splits voice semantics across packages;
  - conflicts with ADR 0006's package-owned semantic model.

References:

- `abstractvoice/adapters/base.py`
- `abstractvoice/vm/tts_mixin.py`
- `abstractvoice/examples/web_ui.py`
- `abstractvoice/integrations/abstractcore_plugin.py`
- `docs/adr/0006_voice_overlays_are_package_owned_not_generic_adapters.md`
- https://huggingface.co/ScenemaAI/scenema-audio
- https://huggingface.co/ResembleAI/Dramabox

---

## Decision

**Proposed direction**: create a package-owned directed speech contract above
the raw adapter layer before promoting any Scenema-class runtime work.

The likely shape is:

- `SpeechRequest`
- `SpeechCapabilities`
- optional `SpeechPlan` / `SpeechResult` metadata

Candidate `SpeechRequest` fields:

- `text`
- `language`
- `provider`
- `model`
- `profile`
- `voice`
- `instructions`
- `pace`
- `target_duration_s`
- `actions`
- `scene_context`
- `ambient_audio`
- `background_sfx`
- `quality_preset`
- `output_channels`

Each field should be reportable per engine as:

- `native`
- `emulated`
- `unsupported`

Existing APIs should compile into this contract internally instead of exposing
the new structure as a breaking change.

**Why**:

- It keeps AbstractVoice, not individual engines, in charge of user-facing
  semantics.
- It lets simple engines remain simple while still making complex engines
  expressible.
- It provides the missing seam for future runtime rewrites and capability-aware
  integrations.

---

## Dependencies

- **ADRs**:
  - `docs/adr/0005_torch_device_and_dtype_policy.md`
  - `docs/adr/0006_voice_overlays_are_package_owned_not_generic_adapters.md`
- **Backlog tasks**:
  - Planned: `docs/backlog/planned/036_voice_profile_abstraction.md`
  - Proposed: `docs/backlog/proposed/042_capability_residency_hooks.md`
  - Proposed: `docs/backlog/proposed/043_dramabox_nonvendored_feasibility.md`
  - Proposed: `docs/backlog/proposed/scenema/045_shared_runtime_foundation_for_advanced_speech_engines.md`
  - Proposed: `docs/backlog/proposed/scenema/048_abstractvoice_owned_speech_planning_boundary.md`

---

## Suggested implementation if promoted

- Define a minimal internal request/result dataclass or typed dict.
- Add capability-reporting helpers that map fields to
  `native|emulated|unsupported`.
- Compile today's `speak`, `speak_to_bytes`, and plugin request paths into that
  new contract without changing the public API.
- Add one nontrivial engine exercise that uses at least one richer field such as
  `pace` or `target_duration_s`.
- Write or update the ADR before closing the item.

---

## Promotion criteria

- At least one engine integration clearly needs more than the current
  `text + voice + instructions + speed` surface.
- The owning ADR is accepted.
- Field naming and ownership do not conflict with ADR 0006.
- Capability degradation rules are explicit enough that integrations can
  surface them honestly.

---

## Validation ideas

- Unit tests proving legacy API calls compile into the new request structure
  without behavior regressions.
- Capability-reporting tests for one native and one unsupported field.
- Plugin tests proving the dict/JSON surface can forward richer fields without
  redefining them.
- Documentation checks for `api.md`, `architecture.md`, and example server
  request docs.

---

## Non-goals

- This proposal does not authorize immediate Scenema or DramaBox integration.
- This proposal does not move planning semantics into AbstractCore.
- This proposal does not promise that every engine will support every field.

---

## Guidance for future agents

- Do not promote a Scenema-class runtime before agreeing on this contract.
- Keep the first version small and honest; avoid adding speculative fields with
  no engine or UX consumer.
- If the contract creates durable semantics, close the loop with an ADR instead
  of burying policy in implementation notes.
