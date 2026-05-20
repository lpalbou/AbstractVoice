## Task 045: Shared runtime foundation for advanced speech engines

**Date**: 2026-05-20  
**Status**: Proposed  
**Priority**: P1

---

## Main goals

- Define which optional dependencies are acceptable as a shared foundation for
  future advanced local speech engines.
- Minimize future engine integration cost by rewriting orchestration and
  speech-specific glue in AbstractVoice instead of adopting engine-specific
  helper stacks wholesale.
- Preserve the current lightweight base install and explicit-heavy-extra model.

## Secondary goals

- Give future agents a rule for when a dependency belongs in a shared runtime
  layer versus an engine-local adapter.
- Avoid a repeat of "one more engine, one more bespoke stack" as more advanced
  local models appear.

---

## ADR status

- **Governing ADRs**:
  - `docs/adr/0005_torch_device_and_dtype_policy.md`
  - `docs/adr/0006_voice_overlays_are_package_owned_not_generic_adapters.md`
- **ADR impact**: May need a new ADR if this becomes a durable dependency and
  runtime-foundation policy rather than a one-off engine decision.

---

## Context / problem statement

Advanced local speech engines increasingly arrive as bundles of:

- model runtime code;
- prompt compilers;
- chunk planners;
- voice-conversion helpers;
- separator/validator tools;
- bespoke service wrappers.

The maintainership preference for AbstractVoice is stricter:

- keep the base package light;
- avoid vendoring upstream runtimes when possible;
- depend only on a small set of reusable core libraries across engines;
- rewrite engine glue locally when that yields a cleaner package boundary.

Without an explicit shared-runtime policy, each new engine will re-open the same
dependency argument.

---

## Current code reality

- The base package remains intentionally lean in `pyproject.toml`.
- Heavy engines are isolated behind explicit extras and lazy imports.
- `abstractvoice/compute/device.py` and `abstractvoice/compute/dtype.py` already
  provide shared torch policy anchors for multiple engines.
- Existing advanced backends are fragmented:
  - `abstractvoice/audiodit/*`
  - `abstractvoice/omnivoice/*`
  - `abstractvoice/cloning/engine_chroma.py`
- There is no shared runtime layer yet for:
  - prompt compilation
  - duration planning
  - quality validation
  - multi-step speech generation graphs
- The repo already tolerates some engine-specific compromises, but not as a
  preferred pattern:
  - vendored/derived code in `audiodit` and `qwen3_asr`
  - `trust_remote_code=True` in the Chroma cloning path

---

## Constraints

- Keep the base install lightweight.
- Keep heavy optional dependencies behind explicit extras.
- Do not adopt engine-specific helper libraries as shared foundation merely
  because one upstream project uses them.
- Prefer dependencies that can plausibly serve more than one engine family.
- Preserve explicit prefetch and no-surprise-download behavior.

---

## Research, options, and references

### Option A: let each engine bring its full bespoke helper stack

- **Pros**:
  - fastest path to getting a new engine running;
  - minimal up-front architecture work.
- **Cons**:
  - dependency sprawl;
  - duplicated orchestration logic;
  - weak package coherence;
  - poorer Apple/MPS prospects because CUDA-oriented helper stacks leak into
    shared paths.

### Option B: rewrite speech-specific orchestration locally and keep only a small reusable library set

- **Pros**:
  - cleanest long-term architecture;
  - strongest fit with current package values;
  - best chance of cross-engine reuse.
- **Cons**:
  - more up-front engineering work;
  - requires discipline about what belongs in the shared layer.

### Option C: adopt a family-specific shared runtime only when at least two engines justify it

- **Pros**:
  - practical compromise for model families such as LTX-derived engines;
  - avoids reimplementing deep model loaders too early.
- **Cons**:
  - still risks importing a broad surface area if the family runtime is unstable
    or overly general;
  - needs explicit guardrails so family-specific code does not become another
    implicit engine bundle.

References:

- `pyproject.toml`
- `abstractvoice/compute/device.py`
- `abstractvoice/compute/dtype.py`
- `abstractvoice/cloning/engine_chroma.py`
- `docs/backlog/proposed/043_dramabox_nonvendored_feasibility.md`
- https://github.com/ScenemaAI/scenema-audio
- https://github.com/resemble-ai/DramaBox

---

## Decision

**Proposed direction**: prefer a self-owned speech runtime layer built on a
small shared dependency set, with conditional use of a family runtime only when
multiple engines justify it.

Shared-foundation candidates:

- `torch`
- `numpy`
- `soundfile`
- `torchaudio`
- `safetensors`
- `huggingface_hub`
- `transformers`
- `tokenizers`
- `sentencepiece`
- `accelerate`

Optional accelerator-only or engine-local dependencies may still exist, but
they should not define the shared foundation:

- `bitsandbytes`
- `xformers`
- `sageattention`
- engine-local VC/separator stacks

Components that should be rewritten in AbstractVoice when promoted:

- prompt validators and compilers;
- chunk planners and duration heuristics;
- capability mapping and degradation policy;
- request/result compilation;
- post-process orchestration;
- warmup/residency glue.

Family-specific runtimes should be allowed only when all of these are true:

- at least two target engines justify the family;
- the family runtime reduces duplication materially;
- the package API is stable enough for non-vendored use;
- the resulting dependency surface is still explainable.

**Why**:

- This matches the user's stated requirement more closely than wrapping each
  upstream bundle.
- It gives Scenema-class work a realistic technical path without committing to
  full bespoke rewrites of every tensor-level detail on day one.
- It keeps the repo's dependency story coherent.

---

## Dependencies

- **ADRs**:
  - `docs/adr/0005_torch_device_and_dtype_policy.md`
  - `docs/adr/0006_voice_overlays_are_package_owned_not_generic_adapters.md`
- **Backlog tasks**:
  - Planned: `docs/backlog/planned/027_refresh_dependency_check.md`
  - Proposed: `docs/backlog/proposed/041_python310_all_apple_numpy_conflict.md`
  - Proposed: `docs/backlog/proposed/043_dramabox_nonvendored_feasibility.md`
  - Proposed: `docs/backlog/proposed/scenema/044_package_owned_directed_speech_request_and_capabilities.md`
  - Proposed: `docs/backlog/proposed/scenema/046_scenema_class_python_runtime_spike.md`
  - Proposed: `docs/backlog/proposed/scenema/047_device_agnostic_runtime_and_apple_silicon_path.md`

---

## Suggested implementation if promoted

- Document the allowed shared runtime dependency tiers.
- Extract or create a small internal module for shared advanced-speech helpers.
- Move any reusable planning or validation logic into that module instead of
  leaving it engine-local.
- Keep accelerator-only dependencies optional and engine-local.
- Add dependency-check and import tests that prove the base install remains
  lean.

---

## Promotion criteria

- At least one proposed advanced engine would otherwise require a large bespoke
  helper stack.
- There is evidence that more than one engine could reuse the same foundation.
- The dependency matrix remains compatible with the repo's lightweight-base
  policy.
- The required family runtime, if any, has a stable non-vendored installation
  path.

---

## Validation ideas

- Base-install import test proving the shared foundation remains opt-in.
- Extras audit comparing before/after dependency counts and import surfaces.
- One real engine integration spike using only the allowed shared-foundation
  dependencies plus narrowly scoped engine-local accelerators.
- Documentation updates in `docs/dependencies.md` and install-profile docs.

---

## Non-goals

- This proposal does not authorize blindly copying upstream helper repos into
  AbstractVoice.
- This proposal does not require every current engine to be refactored
  immediately.
- This proposal does not make `bitsandbytes` or any CUDA-only accelerator a
  required cross-engine dependency.

---

## Guidance for future agents

- Treat "shared dependency" as a high bar. Reuse must be real, not aspirational.
- Prefer local rewrites for speech-specific control flow even when upstream
  helpers exist.
- If a family runtime becomes necessary, record why a narrower local rewrite was
  not the better trade-off.
