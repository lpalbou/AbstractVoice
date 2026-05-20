## Task 053: DramaBox base-TTS engine and capability rollout

**Date**: 2026-05-20  
**Status**: Proposed  
**Priority**: P2

---

## Context

Base-TTS support is only interesting if the clone-first path succeeds and the
no-reference path is good enough to justify more surface area.

## Current code reality

- the branch now has package-owned `SpeechRequest` and `tts_capabilities`;
- current base TTS adapters remain text-first and the richer directed-speech
  fields are mostly still unsupported;
- `output_channels` and related layout intent exist in the request layer but are
  not wired to an engine yet.

## Problem or opportunity

DramaBox might eventually deserve `tts_engine="dramabox"`, but that should not
be assumed before the runtime and clone path prove out.

## Proposed direction

Only if `052` succeeds, evaluate a base `tts_engine="dramabox"` path with:

- no-reference generation quality check;
- output layout normalization;
- capability rollout only for fields that are actually wired;
- profile/voice semantics that do not overload clone IDs or package-owned
  selectors.

## Why it might matter

- offers a cleaner no-reference UX for the narrow “expressive performance TTS”
  lane if quality justifies it.

## Promotion criteria

- clone-first integration succeeds;
- no-reference output is good enough to stand on its own;
- the resulting engine meaningfully improves a use case not already served well
  by current local options.

## Validation ideas

- compare no-reference samples against existing local engines for the same
  directed prompt;
- verify capability reporting only claims wired fields;
- verify output layout normalization does not break bytes-first APIs.

## Non-goals

- does not become the default local TTS engine;
- does not claim multilingual general-purpose support unless proven;
- does not ship before the clone-first path is solid.

## Guidance for future agents

- keep base-TTS promotion contingent on actual product value, not just on
  runtime feasibility;
- stay conservative with capability claims.
