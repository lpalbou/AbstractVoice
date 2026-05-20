## Task 049: Package-owned expressive prompt mapping and reference-audio conditioning

**Date**: 2026-05-20  
**Status**: Proposed  
**Priority**: P1

---

## Context

DramaBox and Scenema both point to the same missing seam: AbstractVoice can now
carry richer speech intent internally, but it still has no package-owned layer
that compiles that intent into engine-native expressive prompt text or into a
conditioning plan for reference audio.

## Current code reality

- `abstractvoice/speech_request.py` can now carry `instructions`, `pace`,
  `target_duration_s`, `actions`, `scene_context`, `ambient_audio`,
  `background_sfx`, and `output_channels`.
- `abstractvoice/vm/tts_mixin.py` reports these richer fields as unsupported
  rather than pretending they work.
- `abstractvoice/cloning/store.py` and `abstractvoice/cloning/manager.py`
  remain transcript-centric for local cloning flows because existing engines
  often need `reference_text`.
- DramaBox’s most interesting conditioning path appears to use reference audio
  directly, which is a different shape than current OmniVoice/AudioDiT/F5
  expectations.

## Problem or opportunity

Without a package-owned prompt/conditioning planner, each advanced engine will
smuggle expressive semantics into ad hoc strings or local kwargs.

## Proposed direction

Add an internal planning layer that can:

- compile package-owned speech fields into engine-native expressive prompt text;
- separate delivery context from actual ambient/background audio intent;
- represent reference-audio-only conditioning without forcing transcript-first
  semantics where the engine does not need them.

## Why it might matter

- reusable beyond DramaBox;
- likely needed by Scenema-class engines too;
- reduces pressure to widen the public API prematurely.

## Promotion criteria

- at least one runtime spike shows that prompt text and reference-audio
  conditioning materially affect output quality;
- at least two advanced-engine families can reuse the same internal plan shape;
- the planner can stay internal without breaking current public APIs.

## Validation ideas

- planner-unit tests for `SpeechRequest -> DirectedSpeechPlan` compilation;
- compatibility audit proving current public request surfaces remain stable;
- one runtime spike consuming the plan without bespoke integration-layer logic.

## Non-goals

- does not widen the public `VoiceManager` API yet;
- does not guarantee that every engine will support every expressive field;
- does not force transcript requirements onto engines that can work from
  reference audio alone.

## Guidance for future agents

- keep the planner package-owned and internal first;
- avoid inventing DramaBox-only public prompt fields before another engine needs
  the same semantics.
