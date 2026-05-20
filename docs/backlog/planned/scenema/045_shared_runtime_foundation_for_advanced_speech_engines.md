## Task 045: Shared advanced-speech runtime hardening and dependency policy

**Date**: 2026-05-20  
**Status**: Planned  
**Priority**: P1

---

## Main goals

- Harden the shared runtime foundation that now exists for torch-backed
  advanced speech engines.
- Keep the cross-engine dependency story small, explicit, and reusable.
- Extend consistent runtime-info and fallback behavior without turning every
  engine into a bespoke stack.

## Secondary goals

- Preserve the intentionally lean base install and opt-in heavy extras.
- Give future engine work a clearer rule for what belongs in shared runtime
  helpers versus engine-local code.

## Promotion history

- Promoted from
  `docs/backlog/proposed/scenema/045_shared_runtime_foundation_for_advanced_speech_engines.md`
  to `docs/backlog/planned/scenema/045_shared_runtime_foundation_for_advanced_speech_engines.md`
  on 2026-05-20.

---

## ADR status

- **Governing ADRs**:
  - `docs/adr/0005_torch_device_and_dtype_policy.md`
  - `docs/adr/0007_directed_speech_requests_and_planning_are_package_owned.md`
- **ADR impact**: May still need a later ADR update if dependency-tier policy
  hardens beyond the current torch runtime and package-owned planning rules.

---

## Context / problem statement

The repo already proved that advanced engines need a shared torch policy.
What is still open is how much farther that shared layer should go without
turning into an engine family bundle.

Scenema-class work needs:

- device and dtype resolution;
- explicit CPU fallback reporting where allowed;
- consistent runtime-info surfaces;
- a small set of reusable dependencies;
- clear boundaries for what gets rewritten inside AbstractVoice.

Without that discipline, every new heavy engine reopens the same architecture
and dependency argument.

---

## Current code reality

- `abstractvoice/compute/device.py`, `abstractvoice/compute/dtype.py`, and
  `abstractvoice/compute/torch_runtime.py` now provide the shared torch runtime
  policy seam.
- Explicit fallback/runtime reporting is now present in:
  - `abstractvoice/omnivoice/runtime.py`
  - `abstractvoice/audiodit/runtime.py`
  - `abstractvoice/cloning/engine_f5.py`
  - `abstractvoice/adapters/stt_transformers_asr.py`
- The base package remains intentionally lean in `pyproject.toml`, with heavy
  runtimes isolated behind extras and lazy imports.
- Dependency/runtime policy is still only partially codified:
  - some engine-local compromises remain, such as Chroma's remote-code path;
  - not every heavy engine exposes the same runtime-info quality;
  - the allowed shared dependency tiers are still more implicit than explicit.

---

## Constraints

- Keep the base install lightweight.
- Keep heavy runtimes behind explicit extras.
- Do not vendor upstream runtime repos merely to avoid writing glue code.
- Do not let CUDA-only accelerators become mandatory shared dependencies.

---

## Research, options, and references

### Option A: let each advanced engine keep its own runtime conventions

- **Pros**:
  - lowest short-term churn.
- **Cons**:
  - repeated fallback bugs;
  - weak Apple/MPS story;
  - poor runtime introspection consistency.

### Option B: keep a small shared runtime layer and rewrite orchestration locally

- **Pros**:
  - matches current branch direction;
  - keeps dependency policy understandable;
  - improves future Scenema-class work.
- **Cons**:
  - requires continued cleanup of remaining outliers.

**Chosen approach**: Option B.

References:

- `abstractvoice/compute/torch_runtime.py`
- `abstractvoice/omnivoice/runtime.py`
- `abstractvoice/audiodit/runtime.py`
- `abstractvoice/cloning/engine_f5.py`
- `abstractvoice/adapters/stt_transformers_asr.py`
  - `docs/backlog/proposed/dramabox/043_dramabox_nonvendored_feasibility.md`

---

## Decision

Treat the shared torch runtime layer as an established foundation and use this
task to harden and document it rather than to invent it from scratch.

Shared-foundation candidates remain intentionally small:

- `torch`
- `numpy`
- `soundfile`
- `huggingface_hub`
- `transformers`
- `tokenizers`
- `sentencepiece`
- `accelerate`

Engine-local or accelerator-local dependencies may still exist, but they do
not define the shared foundation:

- `bitsandbytes`
- `xformers`
- `sageattention`
- family-specific VC, separator, or validation stacks

---

## Dependencies

- **ADRs**:
  - `docs/adr/0005_torch_device_and_dtype_policy.md`
  - `docs/adr/0007_directed_speech_requests_and_planning_are_package_owned.md`
- **Backlog tasks**:
  - Planned: `docs/backlog/planned/027_refresh_dependency_check.md`
  - Planned: `docs/backlog/planned/scenema/044_package_owned_directed_speech_request_and_capabilities.md`
  - Planned: `docs/backlog/planned/scenema/046_scenema_class_python_runtime_spike.md`
  - Proposed: `docs/backlog/proposed/041_python310_all_apple_numpy_conflict.md`

---

## Implementation plan

- Audit remaining heavy engines for shared runtime-info and fallback parity.
- Document the shared dependency tiers in dependency/install docs.
- Keep package-owned planning, capability mapping, and degradation logic inside
  AbstractVoice rather than in engine helpers.
- Leave accelerator-only behavior engine-local and explicitly optional.
- Add verification that the base install remains lean after each new runtime
  addition.

---

## Success criteria

- Shared runtime resolution is reused across the heavy engine paths that should
  share it.
- Requested/resolved device and fallback state are inspectable where advanced
  runtimes exist.
- The repo has an explicit dependency story for future advanced engines.
- New heavy-engine work no longer needs to re-argue the foundation every time.

---

## Test plan

- `tests/test_compute_runtime_policy.py`
- targeted runtime-info tests for torch-backed engines
- base-install import audit and dependency-count checks
- doc audit for install-profile and dependency guidance

---

## Non-goals

- This task does not require every existing engine to be refactored in one
  pass.
- This task does not authorize broad upstream helper adoption as “shared”
  policy.
- This task does not make CUDA-only accelerators part of the mandatory common
  stack.
