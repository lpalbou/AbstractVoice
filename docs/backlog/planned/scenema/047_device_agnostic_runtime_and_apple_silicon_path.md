## Task 047: Apple Silicon validation and explicit fallback policy for advanced speech runtimes

**Date**: 2026-05-20  
**Status**: Planned  
**Priority**: P2

---

## Main goals

- Validate a realistic Apple Silicon path for advanced speech runtimes after a
  Linux/CUDA proof-of-load exists.
- Make requested/resolved device state and CPU fallback explicit instead of
  silent.
- Keep advanced runtime design device-agnostic in shared layers even when
  first implementation work is Linux/CUDA-first.

## Secondary goals

- Separate “possible on Apple” from “supported on Apple”.
- Keep MPS and CPU downgrade behavior honest for future advanced engines.

## Promotion history

- Promoted from
  `docs/backlog/proposed/scenema/047_device_agnostic_runtime_and_apple_silicon_path.md`
  to `docs/backlog/planned/scenema/047_device_agnostic_runtime_and_apple_silicon_path.md`
  on 2026-05-20.

---

## ADR status

- **Governing ADRs**:
  - `docs/adr/0005_torch_device_and_dtype_policy.md`
  - `docs/adr/0007_directed_speech_requests_and_planning_are_package_owned.md`
- **ADR impact**: May still require an ADR 0005 update if mixed-device or
  downgrade policy becomes more specific across multiple advanced engines.

---

## Context / problem statement

Apple support is now a real branch goal, but it is still the hardest platform
for Scenema-class work. The repo already has the beginnings of explicit
runtime-resolution and fallback policy; what remains is to validate that the
same discipline holds once an actual advanced runtime exists.

The right output of this item is not “Apple magically reaches CUDA parity”.
The right output is an auditable answer about what runs on MPS, what falls back
to CPU, and what remains unsupported.

---

## Current code reality

- `abstractvoice/compute/torch_runtime.py` now provides shared requested versus
  resolved device/dtype resolution with explicit fallback metadata.
- Explicit fallback/runtime reporting now exists in several heavy paths,
  including OmniVoice, AudioDiT, OpenF5, and the Transformers ASR adapter.
- The branch still has no Scenema-class advanced runtime to validate end to
  end on Apple.
- Some external constraints remain:
  - PyTorch MPS is real;
  - Apple support for accelerator extras such as `bitsandbytes` remains mixed;
  - current upstream LTX-family ecosystems are still CUDA-first in practice.

---

## Constraints

- Do not block Linux/CUDA-first runtime spikes on full Apple parity.
- Do not allow hidden CPU fallback for correctness-critical paths.
- Do not make CUDA-only accelerators mandatory shared dependencies.
- Preserve explicit requested/resolved runtime reporting.

---

## Research, options, and references

### Option A: ignore Apple until after an engine ships

- **Pros**:
  - least work today.
- **Cons**:
  - invites CUDA assumptions back into shared layers.

### Option B: validate Apple only after a bounded Linux/CUDA spike exists

- **Pros**:
  - matches branch sequencing;
  - keeps Apple real without stalling first implementation work.
- **Cons**:
  - Apple remains a follow-on instead of an immediate guarantee.

**Chosen approach**: Option B.

References:

- `abstractvoice/compute/torch_runtime.py`
- `docs/adr/0005_torch_device_and_dtype_policy.md`
- `docs/backlog/planned/scenema/046_scenema_class_python_runtime_spike.md`
- https://docs.pytorch.org/docs/2.12/notes/mps.html
- https://huggingface.co/docs/bitsandbytes/en/installation
- https://github.com/Lightricks/LTX-Desktop

---

## Decision

Treat Apple Silicon validation as a separate follow-on item after the
Linux/CUDA proof-of-load.

The expected Apple path is:

1. keep shared planning/runtime layers device-agnostic;
2. validate one minimal advanced runtime path on Apple Silicon;
3. allow explicit CPU fallback only when the runtime reports it clearly;
4. downgrade or reject unsupported features explicitly;
5. only claim Apple support when an end-to-end smoke path exists on real
   hardware.

---

## Dependencies

- **ADRs**:
  - `docs/adr/0005_torch_device_and_dtype_policy.md`
  - `docs/adr/0007_directed_speech_requests_and_planning_are_package_owned.md`
- **Backlog tasks**:
  - Planned: `docs/backlog/planned/scenema/045_shared_runtime_foundation_for_advanced_speech_engines.md`
  - Planned: `docs/backlog/planned/scenema/046_scenema_class_python_runtime_spike.md`
  - Proposed: `docs/backlog/proposed/041_python310_all_apple_numpy_conflict.md`

---

## Implementation plan

- Define the Apple validation matrix for the first advanced runtime:
  - pure MPS path
  - explicit CPU fallback path
  - unsupported accelerators/features
- Add smoke criteria for requested/resolved runtime reporting.
- Verify that unsupported accelerators fail clearly rather than degrading
  silently.
- Document feature downgrades and support boundaries before any “Apple
  supported” claim.

---

## Success criteria

- The first advanced runtime under test can report whether it is on MPS, CPU,
  or an explicit fallback path.
- Unsupported Apple paths fail clearly instead of silently changing behavior.
- Shared runtime layers remain device-agnostic enough that Apple validation is
  a real follow-on, not a redesign.
- Support claims are bounded to what was actually validated on Apple hardware.

---

## Test plan

- Apple Silicon smoke run on real hardware after `046`
- negative tests for unsupported accelerator paths
- runtime-info assertions for requested/resolved device and fallback state
- documentation audit for support claims and downgrade notes

---

## Non-goals

- This task does not promise full CUDA feature parity on Apple.
- This task does not make CPU fallback silent or automatic by default.
- This task does not block the Linux/CUDA spike while no Apple validation
  exists yet.
