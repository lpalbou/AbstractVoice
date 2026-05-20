## Task 052: DramaBox clone-engine integration

**Date**: 2026-05-20  
**Status**: Proposed  
**Priority**: P2

---

## Context

If DramaBox lands at all, the cleanest first fit for current AbstractVoice
architecture is as a cloning engine, not as a base TTS engine.

## Current code reality

- `abstractvoice/cloning/manager.py` already owns clone-store lifecycle,
  prefetch, warmup, and runtime-info flows.
- cloned voices already map naturally onto reference audio bundles on disk.
- current local cloning engines are transcript-centric; DramaBox may relax that
  requirement.

## Problem or opportunity

DramaBox’s reference-audio conditioning maps more naturally onto the clone
subsystem than onto the base TTS adapter surface.

## Proposed direction

If `051` succeeds, integrate DramaBox as `cloning_engine="dramabox"` first.

Expected scope:

- explicit prefetch path;
- local runtime wrapper;
- clone-store reuse;
- runtime-info and unload behavior;
- watermark policy exposure;
- no base-TTS promotion yet.

## Why it might matter

- smallest clean product shape for a heavy expressive engine;
- avoids widening base TTS semantics before the runtime is proven.

## Promotion criteria

- `051` proves a repeatable non-vendored runtime path;
- reference-audio conditioning quality is materially better than current local
  options for the target use case;
- packaging/license burden is acceptable for an experimental extra.

## Validation ideas

- one clone-store roundtrip with explicit prefetch;
- runtime-info and unload tests;
- quality check against current clone engines for the same reference sample.

## Non-goals

- does not add `tts_engine="dramabox"` yet;
- does not promise multilingual or Apple-local support;
- does not hide watermark or licensing constraints.

## Guidance for future agents

- keep this clone-first unless the no-reference path clearly justifies a base
  TTS promotion;
- preserve explicit download and hardware messaging.
