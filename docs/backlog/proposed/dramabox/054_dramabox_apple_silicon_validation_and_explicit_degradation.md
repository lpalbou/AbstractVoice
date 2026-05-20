## Task 054: DramaBox Apple Silicon validation and explicit degradation

**Date**: 2026-05-20  
**Status**: Proposed  
**Priority**: P2

---

## Context

Apple support matters for AbstractVoice, but DramaBox upstream is strongly
CUDA-first and relies on a Gemma 4-bit path that is not a clean Apple story
today.

## Current code reality

- the branch now has explicit requested/resolved device reporting and non-silent
  fallback metadata in the shared torch runtime path;
- current advanced-engine planning already treats Apple as a later validation
  track, not as a prerequisite for a Linux/CUDA spike;
- `bitsandbytes` and adjacent accelerator support on Apple remain uncertain.

## Problem or opportunity

If DramaBox ever works on Apple, it must do so with explicit degradation and
clear support boundaries, not with hidden CPU fallback.

## Proposed direction

Validate an Apple path only after runtime feasibility is proven elsewhere.

Likely first shape:

- CPU Gemma or non-bnb text-encoder path;
- MPS for the DiT/VAE pieces if viable;
- explicit runtime-info when parts stay on CPU;
- explicit unsupported/degraded states when that path is not good enough.

## Why it might matter

- preserves a plausible Mac story without pretending it is near-term;
- reuses the branch’s explicit fallback/reporting work.

## Promotion criteria

- a Linux/CUDA spike succeeds first;
- there is a concrete candidate Apple path to validate;
- fallback/degradation reporting can stay explicit and non-silent.

## Validation ideas

- real Apple Silicon smoke run on supported hardware;
- requested/resolved runtime reporting check;
- negative tests showing unsupported accelerator paths fail clearly.

## Non-goals

- does not promise full parity with Linux/CUDA;
- does not justify hidden CPU fallback;
- does not block earlier DramaBox feasibility work.

## Guidance for future agents

- keep “Apple possible” separate from “Apple supported”;
- if the practical path is CPU-only and too slow, say so explicitly rather than
  widening the support claim.
