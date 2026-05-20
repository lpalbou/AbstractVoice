## Task 044: Package-owned directed speech request rollout and truthful capability export

**Date**: 2026-05-20  
**Status**: Planned  
**Priority**: P1

---

## Main goals

- Finish the rollout of the new package-owned `SpeechRequest` seam without
  breaking the current public `VoiceManager` API.
- Keep directed-speech support honest by exporting `native`, `emulated`,
  `conditional`, and `unsupported` capability states from package-owned code
  rather than from engine-specific guesses.
- Fold the speech-planning ownership work into this item now that ADR 0007 and
  the initial internal request/capability layer already exist.

## Secondary goals

- Keep richer speech semantics owned by AbstractVoice instead of leaking them
  into integrations or adapter-local kwargs.
- Preserve room for Scenema-class and DramaBox-class engines without widening
  the raw adapter contract too early.

## Promotion history

- Promoted from
  `docs/backlog/proposed/scenema/044_package_owned_directed_speech_request_and_capabilities.md`
  to `docs/backlog/planned/scenema/044_package_owned_directed_speech_request_and_capabilities.md`
  on 2026-05-20.
- The separate planning-boundary item `048` was folded into this task on
  2026-05-20 after ADR 0007 and the internal request layer landed on the
  `scenema` branch.

---

## ADR status

- **Governing ADRs**:
  - `docs/adr/0005_torch_device_and_dtype_policy.md`
  - `docs/adr/0006_voice_overlays_are_package_owned_not_generic_adapters.md`
  - `docs/adr/0007_directed_speech_requests_and_planning_are_package_owned.md`
- **ADR impact**: The durable ownership rule is already covered by ADR 0007.
  Remaining work here is rollout, enforcement, and public-surface discipline.

---

## Context / problem statement

AbstractVoice now has the start of the right seam for advanced speech work, but
the rollout is incomplete.

The package needs one authoritative place where richer speech intent is
normalized and one authoritative place where support status is reported. That
is required before any Scenema-class runtime can be added without pushing
engine-specific semantics into plugin payloads, web handlers, or adapter
kwargs.

The main question is no longer whether a package-owned request layer is a good
idea. The question is how far to finish that rollout before exposing richer
fields publicly.

---

## Current code reality

- `abstractvoice/speech_request.py` now defines package-owned
  `SpeechRequest`, `SpeechCapability`, and `SpeechCapabilities`.
- `abstractvoice/vm/tts_mixin.py` now compiles current `speak*` paths into that
  internal request layer and reports package-owned TTS capabilities.
- `abstractvoice/integrations/abstractcore_plugin.py` now exports truthful
  control support plus `speech_request_contract: "speech_request_v1"` instead
  of an all-`supported=true` placeholder.
- ADR 0007 now records that directed-speech requests and planning are
  package-owned and that unsupported fields must not disappear silently.
- The public request surfaces are still intentionally thin:
  - `VoiceManager.speak_to_bytes()` and related methods remain text-first;
  - `abstractvoice/examples/web_ui.py` still exposes a minimal HTTP
    `SpeechRequest`;
  - `abstractvoice/integrations/abstractcore_plugin.py` still interprets some
    request-local behavior by temporarily mutating `VoiceManager` state.

---

## Constraints

- Keep the current public `VoiceManager` API stable on this branch.
- Do not overload `profile`, `voice`, `model`, or `instructions` with new
  hidden meanings.
- Do not claim support for a field merely because `SpeechRequest` can carry it.
- Unsupported directed-speech fields must remain explicitly unsupported until
  the package wires them to real behavior.

---

## Research, options, and references

### Option A: expose the richer request publicly now

- **Pros**:
  - fast visible progress;
  - easier to demonstrate the future shape.
- **Cons**:
  - current public surfaces are not aligned enough yet;
  - risks freezing multiple competing request dialects.

### Option B: keep the richer request internal for now and finish the rollout first

- **Pros**:
  - lowest-risk branch shape;
  - lets the package settle semantics before public expansion;
  - matches ADR 0007 cleanly.
- **Cons**:
  - some richer fields remain internal-only for one more cycle.

**Chosen approach**: Option B.

References:

- `abstractvoice/speech_request.py`
- `abstractvoice/vm/tts_mixin.py`
- `abstractvoice/integrations/abstractcore_plugin.py`
- `abstractvoice/examples/web_ui.py`
- `docs/adr/0007_directed_speech_requests_and_planning_are_package_owned.md`

---

## Decision

Keep `SpeechRequest` v1 internal on this branch and finish the rollout before
expanding public request schemas.

Concretely:

- current public entry points keep their existing signatures;
- internal code continues compiling them into package-owned request objects;
- capability export remains package-owned and explicit;
- richer fields such as `pace`, `target_duration_s`, `actions`,
  `scene_context`, `ambient_audio`, `background_sfx`, and `output_channels`
  stay `unsupported` until the package wires them to real planner or engine
  behavior.

This task now owns both the request rollout and the planning-boundary closure
that used to live in `048`.

---

## Dependencies

- **ADRs**:
  - `docs/adr/0006_voice_overlays_are_package_owned_not_generic_adapters.md`
  - `docs/adr/0007_directed_speech_requests_and_planning_are_package_owned.md`
- **Backlog tasks**:
  - Planned: `docs/backlog/planned/scenema/045_shared_runtime_foundation_for_advanced_speech_engines.md`
  - Planned: `docs/backlog/planned/scenema/046_scenema_class_python_runtime_spike.md`

---

## Implementation plan

- Keep `SpeechRequest` v1 small and package-owned.
- Add or tighten manager-owned helper paths where integrations still need to
  normalize request-local state manually.
- Keep plugin and web surfaces aligned with the package-owned capability map.
- Add tests that cover at least one `native`, one `emulated`, and one
  `conditional`, and one `unsupported` field.
- Update API and architecture docs only when a richer public request surface is
  intentionally promoted.

---

## Success criteria

- Internal speech request normalization is package-owned rather than
  integration-owned.
- Capability export is truthful for current engines and explicitly marks
  unimplemented directed-speech fields as unsupported.
- No new public request dialect is introduced before the internal rollout is
  coherent.
- Planning ownership is no longer tracked as a separate backlog item because it
  is already enforced by ADR 0007 plus this rollout work.

---

## Test plan

- `tests/test_speech_request.py`
- `tests/test_abstractcore_plugin.py`
- targeted `VoiceManager` request-compilation tests
- doc-link audit after removing `048`

---

## Non-goals

- This task does not widen `TTSAdapter` into a full directed-speech adapter
  contract yet.
- This task does not make richer directed-speech fields public just because the
  internal dataclass exists.
- This task does not promise engine behavior that the package still reports as
  unsupported.
