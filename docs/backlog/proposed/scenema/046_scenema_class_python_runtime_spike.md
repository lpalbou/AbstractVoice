## Task 046: Scenema-class Python runtime spike

**Date**: 2026-05-20  
**Status**: Proposed  
**Priority**: P2

---

## Main goals

- Evaluate a Python-first, non-vendored rewrite path for a Scenema-class
  directed speech engine inside AbstractVoice.
- Separate what should be rewritten locally from what can remain thin wrappers
  over broad model/runtime libraries.
- Bound the effort, risk, and likely promotion path before implementation work
  starts.

## Secondary goals

- Treat Scenema as a reference design rather than as a service to wrap forever.
- Keep the resulting work compatible with the shared-runtime policy proposed in
  task 045.

---

## ADR status

- **Governing ADRs**:
  - `docs/adr/0005_torch_device_and_dtype_policy.md`
  - `docs/adr/0006_voice_overlays_are_package_owned_not_generic_adapters.md`
- **ADR impact**: None yet, but promotion should wait for the request-contract
  ADR implied by task 044.

---

## Context / problem statement

Scenema Audio is interesting because it is not just another TTS backend. It
combines:

- expressive speech generation;
- scene and SFX-aware prompting;
- reference-audio voice transfer;
- chunk planning for long-form output;
- validation and patching passes.

That makes it a useful design target for the richer "directed speech"
capabilities the user wants. But its upstream code is a full service stack, not
the kind of clean package AbstractVoice usually wants to depend on directly.

The question is not "can we shell out to Scenema?" The question is whether
AbstractVoice can own the orchestration and still produce a serious engine.

---

## Current code reality

- The current repo has no directed-speech planner or request object above the
  TTS adapter layer.
- Existing local heavy engines already prove that AbstractVoice can host
  optional torch-backed runtime code when the boundary is clean enough.
- Voice cloning already gives the repo:
  - clone storage
  - explicit prefetch/warmup patterns
  - runtime-info and unload flows
- Upstream Scenema is much larger than a typical adapter:
  - compiler
  - validator
  - chunker
  - engine runtime
  - voice conversion
  - vocal separator
  - validation/retry logic
- Upstream inspection showed roughly 3.7k lines inside `src/audio_core` plus a
  FastAPI server wrapper and a CUDA-oriented Docker/runtime stack.

---

## Constraints

- No vendored Scenema repo code.
- Python first; Rust is not the default implementation path for the first
  runtime spike.
- Use only the shared dependency foundation approved for advanced engines.
- Preserve explicit download/prefetch behavior.
- Treat Linux/CUDA as the first realistic target.
- Keep Apple/MPS as a separate promotion track rather than a prerequisite.

---

## Research, options, and references

### Option A: wrap the upstream service and stop there

- **Pros**:
  - fastest path to a demo;
  - lowest immediate engineering effort.
- **Cons**:
  - does not satisfy the package-ownership goal;
  - retains upstream's service-oriented assumptions;
  - does not solve the broader abstraction problem.

### Option B: rewrite only the orchestration and planning layers, keep model-family loading thin

- **Pros**:
  - strongest fit with current goals;
  - realistic for a Python implementation;
  - maximizes future reuse across other advanced engines.
- **Cons**:
  - still substantial work;
  - requires contract work first.

### Option C: full tensor/runtime rewrite including every upstream helper path

- **Pros**:
  - maximum package ownership.
- **Cons**:
  - too large for a first pass;
  - risks reimplementing poorly understood model-family behavior before it is
    needed;
  - makes Rust temptation stronger before the Python path is proven.

References:

- https://github.com/ScenemaAI/scenema-audio
- https://huggingface.co/ScenemaAI/scenema-audio
- `docs/backlog/proposed/scenema/044_package_owned_directed_speech_request_and_capabilities.md`
- `docs/backlog/proposed/scenema/045_shared_runtime_foundation_for_advanced_speech_engines.md`

---

## Decision

**Proposed direction**: if promoted, do a Python-first rewrite spike that treats
Scenema as a reference architecture and keeps AbstractVoice in charge of the
planning/orchestration layers.

The likely phase split is:

### Phase 0: contract and shared foundation

- depends on tasks 044 and 045;
- no engine runtime yet.

### Phase 1: Linux/CUDA proof-of-load

- one prompt to WAV using a thin runtime wrapper;
- no separator, no Apple path, no service wrapper requirement.

### Phase 2: package-owned planning and long-form generation

- prompt compiler
- chunk planner
- chunk stitching
- explicit request/result metadata

### Phase 3: optional advanced passes

- reference-audio voice transfer
- validation/retry loop
- ambient/background handling
- optional cleanup passes

### Phase 4: hardening

- docs
- tests
- runtime-info
- unload/warmup
- integration surfaces

Estimated effort for one experienced maintainer:

- MVP spike: **2 to 4 months**
- solid optional engine: **4 to 6 months**

**Why**:

- This keeps the work large but bounded.
- It avoids pretending a Rust rewrite or Apple parity is a realistic first move.
- It aligns the engine with package-owned semantics instead of upstream service
  assumptions.

---

## Dependencies

- **ADRs**:
  - `docs/adr/0005_torch_device_and_dtype_policy.md`
  - `docs/adr/0006_voice_overlays_are_package_owned_not_generic_adapters.md`
- **Backlog tasks**:
  - Proposed: `docs/backlog/proposed/scenema/044_package_owned_directed_speech_request_and_capabilities.md`
  - Proposed: `docs/backlog/proposed/scenema/045_shared_runtime_foundation_for_advanced_speech_engines.md`
  - Proposed: `docs/backlog/proposed/scenema/047_device_agnostic_runtime_and_apple_silicon_path.md`
  - Proposed: `docs/backlog/proposed/scenema/048_abstractvoice_owned_speech_planning_boundary.md`

---

## Suggested implementation if promoted

- Start with a small Python spike, not a full feature branch.
- Prove one non-vendored generation path on supported Linux/CUDA hardware.
- Rewrite prompt validation/compilation and chunk planning locally.
- Keep post-process components optional until the base path is stable.
- Add a stop/go review before any separator or voice-conversion work.

---

## Promotion criteria

- Tasks 044 and 045 are accepted or substantially settled.
- A maintainer explicitly wants a Scenema-class local engine beyond curiosity.
- Required runtime libraries can be installed non-vendored and documented
  cleanly.
- There is access to suitable Linux/CUDA hardware for repeatable testing.

---

## Validation ideas

- Golden-prompt comparisons against upstream behavior for:
  - plain expressive speech
  - scene/SFX-aware prompt
  - long-form chunked output
- Runtime-memory and cold/warm timing logs.
- Explicit proof that generated audio can flow through `VoiceManager` and plugin
  surfaces without shelling out to an external service.
- Tests that no vendored upstream engine code was added.

---

## Non-goals

- This proposal does not make Rust the initial implementation language.
- This proposal does not promise Apple Silicon support in the first runtime.
- This proposal does not authorize shipping a thin subprocess wrapper as the end
  state.

---

## Guidance for future agents

- Treat this as a reference-design rewrite, not an obligation to mimic every
  upstream implementation detail.
- Keep a sharp line between package-owned orchestration and model-family loading
  code.
- Re-check the dependency story before every promotion step; the runtime should
  not quietly accumulate engine-specific baggage.
