## Task 051: LTX audio runtime spike for DramaBox

**Date**: 2026-05-20  
**Status**: Proposed  
**Priority**: P1

---

## Context

The real blocker for a non-vendored DramaBox path is no longer architectural
hand-wringing. It is whether published upstream LTX packages are sufficient for
the required runtime behavior.

## Current code reality

- AbstractVoice now has a shared torch runtime policy in
  `abstractvoice/compute/torch_runtime.py`.
- Heavy engines already live behind explicit extras and prefetch flows.
- The existing DramaBox feasibility memo found a real package/version gap:
  published `ltx-core` / `ltx-pipelines` are still behind the repo version, and
  published `PromptEncoder` behavior does not obviously match DramaBox’s
  vendored warm/audio-only/4-bit path.

## Problem or opportunity

If the non-vendored LTX-family runtime spike fails, most later DramaBox items
should not promote.

## Proposed direction

Run a tightly bounded runtime spike around:

- published `ltx-core` / `ltx-pipelines` where possible;
- minimal local glue where published APIs are missing;
- one prompt-to-WAV path on supported Linux/CUDA hardware.

Candidate scope:

- warm prompt encoding;
- optional Gemma 4-bit load;
- audio-only embeddings processor;
- audio-only transformer builder;
- reference-latent conditioning.

## Why it might matter

- stop/go gate for the whole track;
- potentially reusable for other LTX-family engines later.

## Promotion criteria

- a real non-vendored prompt-to-WAV path works on supported hardware;
- no copied upstream runtime tree is needed;
- the missing pieces are thin enough to own locally.

## Validation ideas

- cold/warm load timing;
- explicit artifact download and cache reuse;
- one no-reference sample and one reference-conditioned sample;
- recorded failure modes when published LTX packages are insufficient.

## Non-goals

- does not promise a production-ready engine;
- does not promise Apple support;
- does not authorize vendoring the DramaBox repo if the spike fails.

## Guidance for future agents

- treat this as the track’s main stop/go gate;
- if unpublished upstream LTX code becomes necessary, stop and re-evaluate the
  whole non-vendoring premise.
