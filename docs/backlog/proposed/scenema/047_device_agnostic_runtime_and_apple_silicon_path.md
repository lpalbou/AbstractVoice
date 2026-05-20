## Task 047: Device-agnostic runtime and Apple Silicon path for advanced speech engines

**Date**: 2026-05-20  
**Status**: Proposed  
**Priority**: P2

---

## Main goals

- Define a realistic path for running future advanced speech engines on Apple
  Silicon without hard-coding the entire design around CUDA.
- Ensure future runtime work respects the existing shared torch device/dtype
  policy instead of reintroducing ad hoc device logic.
- Separate "possible in principle" from "supported in product".

## Secondary goals

- Keep Linux/CUDA-first engine work compatible with a later MPS or mixed-device
  path.
- Avoid optional dependencies that would make Apple support impossible by
  design.

---

## ADR status

- **Governing ADRs**:
  - `docs/adr/0005_torch_device_and_dtype_policy.md`
- **ADR impact**: May revise ADR 0005 if advanced engines need stronger
  cross-engine rules for mixed-device execution or partial fallback policy.

---

## Context / problem statement

Scenema-class and DramaBox-class runtimes are currently CUDA-first upstream.
That does not prove Apple-local execution is impossible, but it does mean
AbstractVoice would need to design for it explicitly.

The risk is straightforward:

- if a future runtime is authored with CUDA assumptions in its shared layer,
  Apple support becomes an expensive rewrite;
- if Apple is treated as a hard requirement too early, the first engine may
  never land at all.

The right question is not "can MPS run some tensors?" The right question is
"what parts of the runtime can be device-agnostic, what parts may need CPU or
feature downgrades, and when do we call Apple support real?"

---

## Current code reality

- `docs/adr/0005_torch_device_and_dtype_policy.md` already establishes shared
  device/dtype helpers as the starting point for torch-backed engines.
- Existing torch engines in this repo already operate with device-selection
  helpers and engine-local caveats rather than a single global CUDA-only rule.
- There is no advanced LTX-family runtime in-tree yet, so there is still time
  to avoid hard-coding CUDA into shared layers.
- External evidence inspected during research:
  - PyTorch MPS is officially supported on Apple Silicon.
  - `bitsandbytes` support on Apple remains mixed/experimental.
  - current LTX desktop guidance still treats macOS Apple Silicon as API-only
    rather than a first-class local generation target.

---

## Constraints

- Do not block Linux/CUDA-first engine work on full Apple parity.
- Do not make CUDA-only accelerators mandatory in the shared runtime
  foundation.
- Use the shared torch device/dtype helpers as the default starting point.
- Keep capability degradation explicit when a feature is unavailable on MPS or
  CPU.

---

## Research, options, and references

### Option A: accept CUDA-only engines and ignore Apple for now

- **Pros**:
  - simplest short-term path.
- **Cons**:
  - makes later Apple support dramatically harder;
  - conflicts with the repo's broader cross-platform goals.

### Option B: design the shared runtime to be device-agnostic, but only support Linux/CUDA first

- **Pros**:
  - best balance between practicality and future portability;
  - keeps Apple alive as a later track rather than a rewrite.
- **Cons**:
  - requires discipline about mixed-device and optional-accelerator boundaries.

### Option C: require Apple parity before promoting any advanced engine

- **Pros**:
  - strongest cross-platform stance.
- **Cons**:
  - likely stalls advanced-engine work indefinitely;
  - overweights the hardest platform before the first engine even exists.

References:

- `docs/adr/0005_torch_device_and_dtype_policy.md`
- https://docs.pytorch.org/docs/2.12/notes/mps.html
- https://huggingface.co/docs/bitsandbytes/en/installation
- https://github.com/Lightricks/LTX-Desktop
- https://github.com/ScenemaAI/scenema-audio
- https://github.com/resemble-ai/DramaBox

---

## Decision

**Proposed direction**: keep advanced runtime design device-agnostic, but treat
Apple Silicon as a separate promotion track after Linux/CUDA proof-of-load.

Likely Apple path:

1. remove CUDA assumptions from shared planning/runtime layers;
2. get core generation working on `mps` or CPU without advanced accelerators;
3. allow mixed-device execution where text encoding, validation, or post-process
   stages can remain on CPU if needed;
4. add feature downgrades for unsupported components instead of pretending full
   parity;
5. only call Apple support real once an end-to-end smoke path exists on a
   supported Apple Silicon machine.

Expected early trade-offs:

- no unconditional `bitsandbytes` requirement;
- no promise that validation or separator stages work on day one;
- possible CPU text-encoder or post-process fallback;
- slower throughput than Linux/CUDA.

**Why**:

- This keeps Apple possible without letting it dominate the first engine plan.
- It respects ADR 0005 better than a new CUDA-only exception stack would.
- It makes support claims auditable.

---

## Dependencies

- **ADRs**:
  - `docs/adr/0005_torch_device_and_dtype_policy.md`
- **Backlog tasks**:
  - Proposed: `docs/backlog/proposed/041_python310_all_apple_numpy_conflict.md`
  - Proposed: `docs/backlog/proposed/scenema/045_shared_runtime_foundation_for_advanced_speech_engines.md`
  - Proposed: `docs/backlog/proposed/scenema/046_scenema_class_python_runtime_spike.md`

---

## Suggested implementation if promoted

- Add a small device-capability matrix for advanced engines.
- Define where mixed-device execution is acceptable.
- Ban direct `torch.cuda.*` assumptions in shared layers.
- Add Apple-specific smoke criteria after Linux/CUDA proof-of-load exists.
- Keep accelerator-specific code paths optional and well-isolated.

---

## Promotion criteria

- A Linux/CUDA engine path already exists or is close enough that portability
  work is no longer speculative.
- Shared-layer code is already mostly device-agnostic.
- At least one maintainer can validate on real Apple Silicon hardware.
- Feature downgrade rules are explicit and documented.

---

## Validation ideas

- Smoke tests proving device selection flows through shared compute helpers.
- One Apple Silicon proof-of-load run for a minimal advanced engine path.
- Negative tests proving unsupported accelerators fail clearly instead of
  crashing late.
- Runtime info that reports requested and resolved device/dtype for advanced
  engines.

---

## Non-goals

- This proposal does not promise full CUDA feature parity on Apple.
- This proposal does not require Apple support before a Linux/CUDA-first engine
  can be explored.
- This proposal does not turn CPU fallback into a hidden correctness crutch.

---

## Guidance for future agents

- Keep "Apple possible later" separate from "Apple supported now".
- Avoid treating experimental third-party Apple support as a stable product
  dependency.
- If a proposed engine cannot be made device-agnostic in shared layers, record
  that explicitly before promotion.
