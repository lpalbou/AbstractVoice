## Task 048: Keep speech planning package-owned in AbstractVoice

**Date**: 2026-05-20  
**Status**: Proposed  
**Priority**: P1

---

## Main goals

- Keep advanced speech-planning semantics owned by AbstractVoice rather than
  pushing them upward into AbstractCore.
- Prevent dependency inversion as speech requests become richer and more
  text-aware.
- Define the preferred fallback design if more than one package needs the same
  request/result contracts later.

## Secondary goals

- Protect the current thin integration shape for AbstractCore plugins and tools.
- Avoid splitting voice semantics across multiple repos.

---

## ADR status

- **Governing ADRs**:
  - `docs/adr/0006_voice_overlays_are_package_owned_not_generic_adapters.md`
- **ADR impact**: Likely needs a new ADR once a directed-speech request
  contract is promoted.

---

## Context / problem statement

The richer the future engine surface becomes, the easier it is to argue that
"planning" belongs in a higher-level orchestration package such as
AbstractCore. That would be a mistake here.

Advanced speech planning is not generic workflow glue. It is voice-domain
semantics:

- how `pace` should be interpreted;
- what `scene_context` means;
- how `ambient_audio` differs from a performance instruction;
- when a request should degrade versus fail;
- how cloned `voice` and base `profile` interact.

If AbstractCore owns that layer, AbstractVoice becomes a low-level execution
detail for semantics it no longer controls, despite being the package that owns
the voice domain.

---

## Current code reality

- `abstractvoice/integrations/abstractcore.py` provides optional tool wiring
  that forwards requests into `VoiceManager`.
- `abstractvoice/integrations/abstractcore_plugin.py` already uses dict/JSON
  shaped inputs and keeps process-local `VoiceManager` ownership inside
  AbstractVoice.
- `VoiceManager` remains the main package-owned orchestration surface for TTS,
  STT, cloning, profiles, and runtime behavior.
- `docs/adr/0006_voice_overlays_are_package_owned_not_generic_adapters.md`
  already states that package-owned semantics must not be redefined by
  integration layers.
- There is no shared cross-package request-contract package today.

---

## Constraints

- Preserve the intended dependency direction: AbstractVoice may integrate with
  AbstractCore, but must not require AbstractCore to define core voice
  semantics.
- Keep integration surfaces serializable as dict/JSON payloads.
- Allow a future neutral contracts package only if multiple real consumers need
  it.
- Avoid baking voice semantics into plugins that should remain thin.

---

## Research, options, and references

### Option A: move advanced speech planning into AbstractCore

- **Pros**:
  - centralizes cross-capability orchestration at a higher layer.
- **Cons**:
  - reverses package ownership of voice semantics;
  - makes AbstractVoice depend on decisions made elsewhere;
  - increases coupling between packages.

### Option B: keep planning and request compilation in AbstractVoice, expose dict/JSON contracts outward

- **Pros**:
  - preserves voice-domain ownership;
  - keeps plugins thin;
  - matches ADR 0006's package-owned semantic model.
- **Cons**:
  - requires AbstractVoice to define and maintain the richer contract.

### Option C: extract a tiny neutral contracts package only after real reuse appears

- **Pros**:
  - avoids dependency inversion if multiple packages later need the same
    request/result schema.
- **Cons**:
  - premature if only AbstractVoice really owns and uses the semantics;
  - still requires AbstractVoice to define the first version.

References:

- `abstractvoice/integrations/abstractcore.py`
- `abstractvoice/integrations/abstractcore_plugin.py`
- `abstractvoice/vm/manager.py`
- `docs/adr/0006_voice_overlays_are_package_owned_not_generic_adapters.md`
- `docs/backlog/proposed/scenema/044_package_owned_directed_speech_request_and_capabilities.md`

---

## Decision

**Proposed direction**: keep directed-speech planning, request compilation, and
capability degradation semantics inside AbstractVoice.

Integration rule:

- AbstractVoice owns the request model and planning semantics.
- AbstractCore forwards serializable request/response dictionaries and exposes
  tool or plugin surfaces.
- If future reuse goes beyond AbstractCore and one-off servers, extract a small
  neutral contracts package later rather than moving ownership upward now.

The likely architectural split is:

- `SpeechRequest` and planning logic in AbstractVoice
- dict/JSON boundary adapters in integrations
- optional later extraction of a contracts-only package if reuse is proven

**Why**:

- This preserves the voice-domain source of truth.
- It prevents dependency inversion.
- It lets AbstractCore stay generic instead of learning speech semantics it does
  not own.

---

## Dependencies

- **ADRs**:
  - `docs/adr/0006_voice_overlays_are_package_owned_not_generic_adapters.md`
- **Backlog tasks**:
  - Proposed: `docs/backlog/proposed/scenema/044_package_owned_directed_speech_request_and_capabilities.md`
  - Proposed: `docs/backlog/proposed/scenema/045_shared_runtime_foundation_for_advanced_speech_engines.md`
  - Proposed: `docs/backlog/proposed/scenema/046_scenema_class_python_runtime_spike.md`

---

## Suggested implementation if promoted

- Define the request/result contracts in AbstractVoice first.
- Keep plugin and server layers serialization-focused.
- Add an import-graph check or review rule that avoids new AbstractCore-owned
  voice semantics.
- Revisit extraction only if another package becomes a genuine peer consumer of
  the same contract.

---

## Promotion criteria

- The richer directed-speech request contract is accepted or close to settled.
- There is active pressure to push planning into AbstractCore or another
  integration layer.
- More than one integration surface needs the same semantics, but AbstractVoice
  still remains the domain owner.

---

## Validation ideas

- Plugin tests proving richer request payloads can pass through without
  reinterpreting semantics in the plugin layer.
- Import-graph review showing AbstractVoice does not depend on AbstractCore for
  core voice behavior.
- Documentation updates in architecture docs explaining the ownership boundary.

---

## Non-goals

- This proposal does not ban all future contract extraction.
- This proposal does not require AbstractCore to become voice-unaware; it only
  limits ownership of semantics.
- This proposal does not define the full request schema by itself.

---

## Guidance for future agents

- Treat "serializable at the boundary" and "owned by AbstractVoice" as separate
  concerns.
- If a future neutral contracts package is proposed, extract only schemas and
  enums, not planning logic.
- Do not let convenience inside one integration layer redefine package-owned
  voice semantics.
