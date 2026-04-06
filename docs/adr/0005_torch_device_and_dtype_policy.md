## ADR 0005: Torch device + dtype selection policy

**Date**: 2026-04-05  
**Status**: Accepted

---

## Context

AbstractVoice now includes multiple **torch-based** backends (already: cloning engines; soon: additional TTS engines).
These engines must run well across:

- macOS (Apple Silicon) → **MPS**
- Linux/Windows with NVIDIA → **CUDA**
- CPU-only environments → **CPU fallback**

Today we already have a partial policy:

- `abstractvoice/compute/device.py` provides `best_torch_device()` and an env override (`ABSTRACTVOICE_TORCH_DEVICE`).
- Individual engines still make ad-hoc dtype decisions (e.g. fp16 vs bf16), which won’t scale as we add more engines.

We need a single, documented policy so new engines (e.g. LongCat-AudioDiT, OmniVoice) behave consistently by default.

---

## Decision

### 1) Centralize device selection for torch-based engines

- Use `best_torch_device()` as the **single source of truth** for automatic device selection.
- Default priority (when not overridden): **cuda → mps → xpu → cpu**
- Allow explicit override via `ABSTRACTVOICE_TORCH_DEVICE` (e.g. `cpu`, `mps`, `cuda`).

### 2) Centralize dtype selection (torch) with safe defaults + override

Add a shared helper (or extend `abstractvoice/compute/*`) so engines can request a recommended dtype:

- **CUDA**: prefer **bf16** when supported; otherwise **fp16**
- **MPS**: default **fp16**; fall back to **fp32** when kernels are unstable
- **CPU**: **fp32**

Allow override via `ABSTRACTVOICE_TORCH_DTYPE` (e.g. `float32`, `float16`, `bfloat16`) so advanced users can force behavior.

### 3) Reliability-first fallbacks

Engines should implement best-effort fallbacks on failure (especially on MPS):

- If an operation fails due to dtype/device incompatibility, retry on **CPU fp32** where feasible.
- If a fallback is taken, emit an explicit `#FALLBACK`-style warning (consistent with repo conventions).

### 4) Keep device/dtype policy independent from model download policy

Device/dtype selection must not change offline-first behavior:

- Large model downloads remain gated by `allow_downloads` and/or explicit prefetch commands.

---

## Consequences

### Positive

- Consistent performance defaults across engines and platforms.
- Clear, centralized override knobs for integrators.
- Reduced per-engine “special-case” logic as new TTS/cloning backends are added.

### Negative / Risks

- Any default policy will be wrong for some models; per-engine overrides may still be needed.
- Some models may require CPU fallbacks on MPS for correctness, reducing performance.

---

## Related

- `abstractvoice/compute/device.py` (current device selection)
- `abstractvoice/compute/dtype.py` (dtype defaults + `ABSTRACTVOICE_TORCH_DTYPE`)
- `docs/adr/0004_streaming_and_cancellation_for_cloned_tts.md`
