## Task 050: Advanced speech output layout and result metadata

**Date**: 2026-05-20  
**Status**: Proposed  
**Priority**: P1

---

## Context

DramaBox is stereo-oriented and upstream treats output layout, sample rate, and
optional watermarking as first-class runtime properties. AbstractVoice still
returns audio primarily as bytes with lightweight metrics.

## Current code reality

- `VoiceManager.speak_to_bytes()` is still bytes-first.
- TTS metrics now include additive runtime fields such as
  `request_contract: "speech_request_v1"`.
- `SpeechRequest` already has `output_channels`, but current capabilities mark
  it unsupported and the current result surface does not express layout or
  watermark state cleanly.

## Problem or opportunity

Advanced engines need more output/result metadata than “here are the bytes”.

## Proposed direction

Add an internal result/output metadata layer that can capture:

- sample rate;
- channel layout;
- requested versus resolved output layout;
- watermark present/disabled/unknown state;
- runtime fallback state where relevant.

Keep current bytes-returning public APIs compatible by treating the richer
result metadata as additive.

## Why it might matter

- reusable for DramaBox, Scenema-class engines, and any future stereo or
  scene-aware backends;
- keeps compatibility while making advanced runtime behavior inspectable.

## Promotion criteria

- at least one advanced runtime produces materially different output layouts or
  watermark states worth exposing;
- the additive metadata shape can be introduced without breaking existing
  callers that only consume bytes.

## Validation ideas

- metadata-only tests around `pop_last_tts_metrics()` and catalog/runtime-info
  surfaces;
- one smoke path proving stereo/high-rate output can flow through current
  managers without schema breakage.

## Non-goals

- does not require immediate public API redesign;
- does not make watermark policy final by itself.

## Guidance for future agents

- keep legacy bytes-returning calls intact;
- prefer additive metadata over a parallel result API unless multiple runtimes
  prove the richer object is necessary.
