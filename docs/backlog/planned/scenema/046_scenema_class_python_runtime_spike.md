## Task 046: Scenema-class Linux/CUDA proof-of-load spike

**Date**: 2026-05-20  
**Status**: Planned  
**Priority**: P2

---

## Main goals

- Prove one non-vendored Scenema-class prompt-to-WAV path inside
  AbstractVoice on realistic Linux/CUDA hardware.
- Exercise the new package-owned request/capability seam with at least one
  richer field instead of using a thin subprocess or service wrapper.
- Bound the runtime effort before any larger Scenema-class rewrite is
  promoted.

## Secondary goals

- Treat Scenema as a reference architecture, not as a repo to mirror.
- Gather enough memory, latency, and complexity evidence to decide whether the
  engine should advance beyond the spike.

## Promotion history

- Promoted from
  `docs/backlog/proposed/scenema/046_scenema_class_python_runtime_spike.md` to
  `docs/backlog/planned/scenema/046_scenema_class_python_runtime_spike.md` on
  2026-05-20.

---

## ADR status

- **Governing ADRs**:
  - `docs/adr/0005_torch_device_and_dtype_policy.md`
  - `docs/adr/0007_directed_speech_requests_and_planning_are_package_owned.md`
- **ADR impact**: None yet. This item depends on the already-accepted request
  ownership rule rather than creating a new one.

---

## Context / problem statement

The branch now has the beginnings of the right architecture for a Scenema-class
engine, but there is still no actual advanced directed-speech runtime in-tree.

The next useful question is not whether a full rewrite is theoretically
possible. It is whether a bounded proof-of-load can run through the new package
seams without vendoring Scenema and without committing to the entire upstream
feature surface.

---

## Current code reality

- Internal package-owned request and capability helpers now exist.
- Shared torch runtime resolution and explicit fallback reporting now exist.
- Existing heavy engines already prove that the repo can host optional
  torch-backed runtime code behind extras.
- No Scenema-class runtime exists yet:
  - no scene-aware prompt compiler wired to generation;
  - no chunk-planning/runtime path for a directed-speech engine;
  - no minimal LTX-family prompt-to-WAV adapter in-tree.

---

## Constraints

- No vendored Scenema repo code.
- Python first.
- Linux/CUDA first for the spike.
- Use the shared runtime and package-owned request seams rather than bypassing
  them.
- Keep Apple/MPS support out of scope for this spike; it is tracked in `047`.

---

## Research, options, and references

### Option A: wrap the upstream service and stop there

- **Pros**:
  - fast demo.
- **Cons**:
  - wrong package boundary;
  - weak long-term value.

### Option B: bounded in-process proof-of-load using package-owned seams

- **Pros**:
  - tests the architecture we actually want;
  - still small enough to de-risk first.
- **Cons**:
  - requires more local runtime glue than a wrapper.

**Chosen approach**: Option B.

References:

- https://github.com/ScenemaAI/scenema-audio
- https://huggingface.co/ScenemaAI/scenema-audio
- `docs/backlog/planned/scenema/044_package_owned_directed_speech_request_and_capabilities.md`
- `docs/backlog/planned/scenema/045_shared_runtime_foundation_for_advanced_speech_engines.md`

---

## Decision

Keep this item tightly bounded to a Linux/CUDA proof-of-load spike.

The spike should prove:

- one in-process generation path;
- one package-owned request normalization path;
- one truthful capability story for the richer field exercised;
- one documented install/runtime envelope.

It should not absorb long-form stitching, separator work, voice conversion, or
Apple portability in the same item.

---

## Dependencies

- **ADRs**:
  - `docs/adr/0005_torch_device_and_dtype_policy.md`
  - `docs/adr/0007_directed_speech_requests_and_planning_are_package_owned.md`
- **Backlog tasks**:
  - Planned: `docs/backlog/planned/scenema/044_package_owned_directed_speech_request_and_capabilities.md`
  - Planned: `docs/backlog/planned/scenema/045_shared_runtime_foundation_for_advanced_speech_engines.md`
  - Planned: `docs/backlog/planned/scenema/047_device_agnostic_runtime_and_apple_silicon_path.md`

---

## Implementation plan

- Choose the smallest viable runtime slice for one Scenema-class prompt-to-WAV
  path.
- Route the request through the package-owned `SpeechRequest` layer.
- Exercise at least one richer field beyond plain text, such as
  `scene_context`, `actions`, or `output_channels`.
- Capture runtime-info, hardware requirements, and cold/warm timing.
- End with a stop/go review before any broader feature work is promoted.

---

## Success criteria

- A Linux/CUDA proof-of-load exists for one Scenema-class generation path
  inside AbstractVoice.
- The spike uses package-owned request/capability seams rather than a thin
  external wrapper.
- Install cost, runtime cost, and complexity are documented well enough to
  decide whether further Scenema-class work is justified.
- Scope stays bounded to the spike.

---

## Test plan

- one reproducible prompt-to-WAV smoke test on supported hardware
- runtime-info capture for requested/resolved device and timing
- negative test or audit confirming no vendored Scenema code was introduced
- integration proof that the output can flow through `VoiceManager`

---

## Non-goals

- This task does not promise Apple Silicon support.
- This task does not include separator, VC, or validation-loop parity.
- This task does not authorize a long-lived subprocess or service wrapper as
  the end state.
