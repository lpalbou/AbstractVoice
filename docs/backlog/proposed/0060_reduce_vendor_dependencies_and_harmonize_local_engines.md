## Task 060: Reduce heavy vendor dependencies (harmonize + lighten AbstractVoice)

**Date**: 2026-05-22  
**Status**: Proposed  
**Priority**: P1

---

## Context

AbstractVoice aims to be **remote-first** and **import-light** in the base install, while keeping
local engines available behind explicit extras. Today, even when the base install stays small, our
local-engine story still pulls in **vendor engine packages** (for example `omnivoice` and
`f5-tts`) that:

- expand the dependency surface area and increase version-conflict risk;
- make installs heavier than necessary for users who only want “low level” stacks (`torch`,
  `transformers`, `onnxruntime`, `soundfile`, etc.);
- make it harder to keep **package-owned abstractions** consistent across the Abstract ecosystem.

`../abstractmusic/` is a useful reference point: it keeps the base package dependency-free and
adds local runtimes via extras that primarily depend on **low-level libraries**, not “engine SDK”
packages.

---

## Current code reality

- Base deps are intentionally small (`pyproject.toml`): `numpy`, `requests`, `appdirs`.
- Optional extras currently include **vendor engine packages**:
  - `omnivoice` extra includes `omnivoice>=0.1.5` and related torch/transformers deps.
  - `cloning` (and `apple`/`gpu`/`all-apple`) include `f5-tts>=1.1.0`.
- Local runtime code imports vendor packages directly:
  - `abstractvoice/omnivoice/runtime.py` loads via `from omnivoice import OmniVoice`.
  - `abstractvoice/cloning/engine_f5.py` imports and executes `f5_tts.*` utilities.
- AbstractVoice already has “import-light” discipline tests and dependency checks (ex:
  `tests/test_lightweight_import_boundaries.py`, `abstractvoice/dependency_check.py`), but those
  do not solve the broader goal of **removing vendor package dependencies** from local-engine
  implementations.
- In contrast, `../abstractmusic/pyproject.toml` keeps base deps empty and avoids depending on
  upstream “engine packages” for model runtimes; it mainly depends on low-level runtime libraries
  in opt-in extras.

---

## Problem or opportunity

We want AbstractVoice’s local engine implementations to be:

- more **package-owned** (our abstractions, our normalization, our request/response surface);
- more **consistent** across engines (harmonized treatment of model IDs, devices, caching, warmup,
  and output metadata);
- **lighter** and easier to install by default (fewer large/high-level vendor dependencies).

This is not about removing “profiles” like `all-apple`; it’s about eliminating (or drastically
reducing) dependencies on vendor engine packages so the extras depend mostly on low-level
libraries.

---

## Proposed direction

1. **Dependency audit and classification**
   - Build a short list of “vendor packages” we want to remove or isolate (initial suspects:
     `omnivoice`, `f5-tts`, possibly others like `piper-tts` / `faster-whisper` if we decide they
     count as vendor rather than low-level).
   - For each, record:
     - why we depend on it today (what code paths / features it provides),
     - what low-level stack could replace it (`torch`, `transformers`, `onnxruntime`, etc.),
     - licensing constraints and feasibility.

2. **Choose a migration strategy (one of)**
   - **Strategy A (preferred)**: re-implement minimal runtime wrappers *package-owned* on top of
     low-level libraries + HF snapshots (use vendor repos as reference, not as runtime deps).
   - **Strategy B**: split vendor-backed engines into separate distributions
     (ex: `abstractvoice-omnivoice`, `abstractvoice-f5`) and keep `abstractvoice` core vendor-free.
   - **Strategy C**: keep vendor packages but harden import boundaries + move them behind explicit
     plugin entry points (still heavier; probably only a stopgap).

3. **Apply to the biggest wins first**
   - Start with one engine where vendor dep removal is realistic and high value:
     - OmniVoice TTS/cloning, or F5 cloning.
   - Define the minimal capability subset needed to keep AbstractVoice’s public contract stable.

4. **Packaging + docs**
   - Update `pyproject.toml` extras so “local” installs depend primarily on low-level libraries.
   - Update docs (`docs/installation.md`, `docs/api.md`, `llms*.txt`) to clarify what installs what
     and which features require which low-level stacks.

---

## Why it might matter

- Fewer dependency conflicts (especially around `torch` / `torchvision` / `transformers`).
- Faster installs, smaller wheels, fewer transitive deps.
- More consistent engine behavior (warmup, caching, device selection, error reporting).
- Better long-term maintainability across Abstract packages (similar approach as AbstractMusic).

---

## Promotion criteria

Promote to `planned/` once we have:

- a concrete list of vendor packages to remove/isolate (and explicit non-goals);
- one chosen strategy (A/B/C) with a realistic first engine target and a step plan;
- an initial feasibility spike plan that includes success/failure signals and fallback options.

---

## Validation ideas

- `python -m pip install .` and `python -m pip install ".[<chosen extra>]"` dependency tree checks
  before/after.
- Import-time smoke: `python -c "import abstractvoice; print('ok')"` on base install.
- Lightweight boundaries: keep/extend `tests/test_lightweight_import_boundaries.py` so base import
  remains vendor-free.
- If Strategy B: plugin discovery tests proving the base package runs without vendor engines
  installed.

---

## Non-goals

- No behavior rewrite of the public `VoiceManager` contract in this proposal.
- No assumption that *all* heavy dependencies can be removed (low-level stacks like `torch` may
  remain required for certain local engines).
- No commitment (yet) to removing a specific engine; this item is about deciding the best, least
  disruptive path and sequencing.

---

## Guidance for future agents

Keep changes incremental. Prefer one “pilot” engine conversion first, and treat packaging changes
as user-facing API: document and test the install story as carefully as the runtime behavior.
