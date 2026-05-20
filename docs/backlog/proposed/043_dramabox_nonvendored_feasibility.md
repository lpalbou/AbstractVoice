## Task 043: Evaluate non-vendored DramaBox integration feasibility

**Date**: 2026-05-20  
**Status**: Proposed  
**Priority**: P2

---

## Main Goals

- Determine whether `ResembleAI/Dramabox` can be integrated into AbstractVoice **without vendoring upstream runtime/model code**.
- Bound the likely engineering effort for a clean integration against existing `VoiceManager` / adapter / cloning abstractions.
- Record the success conditions and likely failure modes before any implementation starts.

## Secondary Goals

- Preserve the project preference for explicit downloads, minimal surprise runtime behavior, and clean package boundaries.
- Document the pros and cons so a later agent can decide whether to promote this into a real implementation task.

---

## Context / Problem Statement

DramaBox is attractive because it combines:

- prompt-directed expressive TTS;
- optional voice-reference conditioning for timbre cloning;
- a stronger “performance” surface than the current local engines.

However, the upstream project does **not** ship as a clean Python SDK. The public release is a repository with:

- custom inference scripts;
- a copied `ltx2/` runtime tree;
- pinned CUDA-heavy dependencies;
- large model artifacts;
- a default Gemma 3 12B 4-bit text encoder path.

The maintainership preference for AbstractVoice is to **avoid vendoring external code when possible**. That makes the key question:

> Can we write the integration ourselves against upstream packages and our own abstractions, without copying DramaBox internals into this repo?

---

## Current Code Reality

- AbstractVoice base/local TTS engines are selected through `abstractvoice/adapters/tts_registry.py`.
- Optional cloning engines are loaded lazily through `abstractvoice/cloning/manager.py`.
- Heavy engines already live behind explicit extras in `pyproject.toml` and explicit prefetch flows in:
  - `abstractvoice/prefetch.py`
  - `abstractvoice/__main__.py`
- The REPL and web flows are intentionally offline-first and should not trigger surprise multi-GB downloads during normal use.
- The repo already includes **derived/vendored model code** in two places when that removed a worse integration burden:
  - `abstractvoice/audiodit/*`
  - `abstractvoice/qwen3_asr/*`
- The repo also already tolerates a less ideal external-model path in one place:
  - `abstractvoice/cloning/engine_chroma.py` uses `trust_remote_code=True`.

DramaBox-specific findings from local inspection:

- Upstream ships no `dramabox` package on PyPI.
- Upstream runtime is centered around:
  - `src/inference.py`
  - `src/inference_server.py`
  - `src/model_downloader.py`
  - `src/audio_conditioning.py`
- The upstream runtime depends heavily on LTX components, but those LTX pieces now also exist as standalone upstream packages:
  - `ltx-core`
  - `ltx-pipelines`
- Important caveat: local inspection showed version skew between the upstream LTX repo and published packages:
  - LTX repo metadata currently says `1.1.3`
  - `pip index versions ltx-core` and `ltx-pipelines` currently expose `1.0.0`
- DramaBox runtime relies on custom features that are **not** a trivial one-line use of the published LTX helpers, including:
  - a warm/cached prompt encoder path;
  - `bitsandbytes` 4-bit Gemma loading;
  - audio-only stripping of video text-embedding components;
  - custom audio reference conditioning behavior;
  - custom audio-only transformer builder logic;
  - prompt duration heuristics and laughter/pause budgeting;
  - CFG-aware rescale heuristics;
  - a latent silence-boundary patch for long outputs;
  - optional watermark application.

Operational/product constraints from upstream material:

- Language: English-first.
- Hardware: effectively CUDA/Linux first; upstream reports roughly 24 GB peak VRAM and H100-class warm-server expectations.
- License: LTX-2 Community License, not MIT/Apache-style permissive packaging.

---

## Constraints

- Do **not** vendor the DramaBox repo or the copied upstream `ltx2/` tree into AbstractVoice.
- Keep the public `VoiceManager` contract stable.
- Keep the integration behind an explicit optional extra.
- Preserve explicit-prefetch and no-surprise-download behavior.
- Prefer not to introduce a new large `trust_remote_code` dependency path unless a separate decision explicitly accepts it.
- Treat Linux/CUDA-only support as acceptable for an experimental extra, but do not pretend this is an Apple/local-desktop general engine.
- Keep model/license caveats explicit in docs.

---

## Research, Options, and References

### Option A: Vendor DramaBox or its copied LTX runtime

- **Summary**: Copy upstream runtime code into AbstractVoice and adapt it locally.
- **Pros**:
  - Fastest path to a technically working integration.
  - Lowest uncertainty about reproducing upstream inference behavior.
- **Cons**:
  - Conflicts with current maintainership preference.
  - High long-term merge/maintenance burden.
  - Imports a large body of external code into this repo.
  - Makes future upstream security/behavior audits more expensive.
- **Status**: Rejected for now unless every thinner option fails and the feature is strategically important enough to justify the cost.

References:
- `docs/development.md`
- `docs/dependencies.md`
- `third_party_licenses/README.md`

### Option B: Wrap the DramaBox repo as an external subprocess or sidecar environment

- **Summary**: Treat upstream DramaBox as a separate installed tool/runtime and call it from AbstractVoice.
- **Pros**:
  - No vendored code in this repo.
  - Minimal reimplementation effort.
- **Cons**:
  - Weakest integration quality.
  - Extra environment management burden for users.
  - Harder to align with `VoiceManager` lifecycle, preload, errors, and portability.
  - Makes tests, packaging, and support much messier.
  - Feels closer to “call a demo script” than to a first-class engine.
- **Status**: Possible as a temporary spike, but not a good long-term AbstractVoice engine design.

References:
- `/tmp/DramaBox/README.md`
- `/tmp/DramaBox/src/inference.py`
- `/tmp/DramaBox/src/inference_server.py`

### Option C: Depend on upstream `ltx-core` + `ltx-pipelines`, write an AbstractVoice-native DramaBox runtime

- **Summary**: Use official upstream LTX packages as dependencies, then write local glue code for the DramaBox-specific inference path.
- **Pros**:
  - Best non-vendored fit if upstream package APIs are good enough.
  - Keeps the heavy/model-specific behavior in one local runtime wrapper.
  - Preserves direct integration with AbstractVoice adapters/cloning abstractions.
- **Cons**:
  - Still requires meaningful custom runtime work.
  - LTX package version skew is a real risk.
  - `ltx-pipelines` pulls a fairly broad dependency surface for an audio-only use case.
  - Upstream `PromptEncoder` does not obviously expose DramaBox’s warm/audio-only/4-bit path directly.
- **Status**: Best current candidate.

References:
- `/tmp/LTX-2/packages/ltx-core/pyproject.toml`
- `/tmp/LTX-2/packages/ltx-pipelines/pyproject.toml`
- `/tmp/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/utils/blocks.py`
- `/tmp/DramaBox/src/inference_server.py`
- `/tmp/DramaBox/src/audio_conditioning.py`

### Option D: Depend on `ltx-core` only, reimplement the minimum needed wrappers locally

- **Summary**: Avoid `ltx-pipelines` and write local equivalents for prompt encoding, audio conditioning, audio decode, and warm model lifecycle.
- **Pros**:
  - Cleanest long-term architecture.
  - Smallest external dependency surface.
  - Avoids carrying unrelated `ltx-pipelines` dependencies for an audio-only engine.
- **Cons**:
  - Highest initial implementation effort.
  - More behavior to validate against upstream DramaBox.
  - Greater risk of subtle inference drift.
- **Status**: Good fallback only if `ltx-pipelines` proves too unstable or too heavy.

References:
- `/tmp/LTX-2/packages/ltx-core/src/ltx_core/*`
- `/tmp/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/utils/blocks.py`

---

## Decision

**Proposed direction**: do **not** start a real engine integration yet. If this idea is promoted, begin with a **proof-of-load spike** using upstream `ltx-core` plus `ltx-pipelines`, with no vendored DramaBox code.

The spike should answer one narrow question first:

> Can published upstream LTX packages, plus a thin local runtime wrapper, load DramaBox checkpoints and produce a repeatable WAV on supported CUDA hardware without copying upstream runtime code?

If the answer is “no” because of version skew or missing APIs, stop and re-evaluate before writing a larger adapter.

**Preferred product shape if the spike succeeds**:

- first integrate as a **cloning engine** (`cloning_engine="dramabox"`) because the existing clone store already carries reference audio and voice lifecycle;
- only evaluate a base `tts_engine="dramabox"` path later if the no-reference and fixed-profile UX is worth the extra surface.

**Why**:

- This is the cleanest path that respects the non-vendoring preference.
- It matches AbstractVoice’s existing optional-heavy-engine pattern.
- It avoids prematurely committing to a large maintenance burden before we know whether upstream packages are stable enough.

---

## Pros and Cons

### Pros

- Strongest candidate in this class for **expressive, prompt-directed performance speech**.
- Better fit than current local engines for theatrical/character/dialogue use cases.
- Optional voice-reference conditioning maps naturally onto the existing clone-store concept.
- Likely avoids the `reference_text` dependency that current local clone engines often need, because the upstream path conditions directly on reference audio.
- Could be valuable as a clearly-labeled experimental CUDA-only extra even if it never becomes a default engine.

### Cons

- Narrower fit than current defaults: not a general-purpose local assistant TTS engine.
- English-first and not a serious replacement for current multilingual paths.
- Very heavy runtime and artifact footprint.
- Operationally much closer to `chroma` than to `piper`/`supertonic`.
- Upstream is repo-centric, not SDK-centric.
- Published LTX package lag raises a real packaging risk.
- Warm runtime behavior depends on custom prompt-encoder logic that we would need to reproduce ourselves.
- License terms are materially less convenient than MIT/Apache-style engines.

---

## Estimated Effort

These are order-of-magnitude estimates for a maintainer with appropriate Linux/CUDA hardware already available.

### Phase 0: proof-of-load spike

- Rough effort: **2 to 4 engineer-days**
- Goal:
  - install/resolve upstream LTX dependencies cleanly;
  - download DramaBox + Gemma artifacts explicitly;
  - prove one successful local inference call from a thin local wrapper.

### Phase 1: minimal AbstractVoice-native experimental engine

- Rough effort: **1 to 2 engineer-weeks**
- Scope:
  - local runtime wrapper;
  - explicit prefetch path;
  - single-engine synthesis to WAV bytes;
  - failure UX and cache handling;
  - no streaming guarantees;
  - no Apple path;
  - no vendored upstream runtime code.

### Phase 2: production-quality optional extra

- Rough effort: **2 to 4 additional engineer-weeks**
- Scope:
  - clone-store integration;
  - preload/unload behavior;
  - tests and docs;
  - plugin/catalog exposure;
  - packaging hardening;
  - clear license/deployment guidance.

### Overall assessment

- **Technical feasibility**: moderate.
- **Packaging/deployment fit for AbstractVoice**: weak to moderate.
- **Likelihood of success as an experimental Linux/CUDA extra**: reasonable.
- **Likelihood of success as a broadly recommended local engine**: low.

---

## Suggested Implementation Plan

- Run a proof-of-load spike in an isolated Linux/CUDA environment.
- Prefer `ltx-core` + `ltx-pipelines` over copied upstream runtime code.
- Write a tiny local `dramabox/runtime.py` wrapper that does only:
  - artifact resolution;
  - prompt encoding;
  - audio reference encoding;
  - audio-only transformer load;
  - denoising;
  - audio decode to WAV bytes.
- Reproduce only the minimum Dramabox-specific heuristics needed for sane first output:
  - audio-only model config/load path;
  - reference-audio conditioning;
  - duration estimate;
  - guidance parameters;
  - rescale rule;
  - long-output silence fix if still needed.
- If the spike requires unpublished upstream LTX code, a Git dependency, or substantial local code copied from upstream helpers, stop and explicitly revisit whether the non-vendoring rule still holds.

---

## Promotion Criteria

Promote this to `planned/` only if all of the following are true:

- a Linux/CUDA spike produces real speech from a thin local wrapper;
- the wrapper does not require vendoring the DramaBox repo or copied LTX runtime tree;
- dependency installation is repeatable enough for an explicit optional extra;
- the resulting engine quality is meaningfully better than existing local options for the targeted expressive-use-case lane;
- the maintainer is willing to support a CUDA-only, English-first, heavy experimental engine in docs and user support.

---

## Success Criteria

- We know whether a non-vendored integration is realistic before writing a large adapter.
- We have an explicit record of the packaging, runtime, and maintenance trade-offs.
- Future implementation work starts from a specific spike plan instead of redoing the same research.

---

## Test Plan

Before promotion:

- Verify dependency resolution on the intended target platform.
- Prove explicit artifact download and cache reuse.
- Generate at least one no-reference sample and one reference-conditioned sample.
- Measure cold-load and warm-request latency.
- Record peak memory behavior and failure modes.
- Confirm whether the upstream package versions on PyPI are sufficient, or whether the spike quietly relied on a repo checkout / unpublished code.

---

## Guidance for Future Agents

- Re-check upstream LTX package publishing status before implementation; the package-lag finding here may become stale.
- Re-check DramaBox license terms before committing implementation work.
- Do not assume Apple/MPS support.
- Do not start by copying upstream helper files into AbstractVoice unless a separate decision explicitly authorizes that tradeoff.
