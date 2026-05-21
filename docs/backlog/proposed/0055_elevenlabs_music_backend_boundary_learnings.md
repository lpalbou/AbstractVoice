## Task 055: ElevenLabs Music backend boundary learnings for voice

**Date**: 2026-05-21  
**Status**: Proposed  
**Priority**: P2

---

## Context

AbstractMusic added an ElevenLabs remote backend scoped only to the official Music API. That work
is relevant to AbstractVoice because ElevenLabs is also a voice/TTS provider, but the packages must
not accidentally blur capability boundaries.

## Current code reality

- AbstractMusic now uses a stdlib-only remote client pattern for provider APIs so the base package
  remains lightweight.
- The music backend is intentionally limited to `/v1/music` and `/v1/music/plan`; it does not call
  ElevenLabs TTS, speech-to-speech, voice design, cloning, or voice APIs.
- Runtime credentials are configured with `ELEVENLABS_API_KEY` / `ELEVENLABS_BASE_URL`, which may
  also be useful to voice providers if AbstractVoice later chooses to support ElevenLabs.
- A live AbstractMusic smoke reached ElevenLabs but returned HTTP 402 `limited_access`, meaning the
  key authenticated but the account tier did not include Music API access.

## Problem or opportunity

If AbstractVoice later adds an ElevenLabs provider, it should reuse the lightweight provider-client
lessons without coupling itself to AbstractMusic or assuming that Music API account access implies
Voice API access.

## Proposed direction

Keep any future ElevenLabs voice support in AbstractVoice as its own voice-scoped provider:

- implement only voice/TTS/STT endpoints that belong to AbstractVoice;
- use stdlib HTTP or a minimal internal adapter first, not the full SDK by default;
- keep generated voice artifacts, request metadata, errors, and account-tier diagnostics explicit;
- avoid importing AbstractMusic or sharing backend classes across packages.

## Why it might matter

- prevents cross-capability leakage between music and voice;
- preserves the lightweight base-package pattern used across the framework;
- lets AbstractVoice handle ElevenLabs voice-specific terms, tiers, models, and errors separately.

## Promotion criteria

- there is an explicit product need for ElevenLabs voice/TTS/STT in AbstractVoice;
- official ElevenLabs voice endpoint behavior, output formats, streaming behavior, and account-tier
  errors are reviewed against current docs;
- the provider can be added without heavy dependencies in the base install.

## Validation ideas

- mocked HTTP tests for request shape, binary output handling, and sanitized error messages;
- opt-in live smoke gated behind `ELEVENLABS_API_KEY`;
- tests proving voice support does not import or depend on AbstractMusic.

## Non-goals

- does not authorize implementing ElevenLabs voice now;
- does not move music generation into AbstractVoice;
- does not create a shared ElevenLabs SDK dependency between packages.

## Guidance for future agents

Use the AbstractMusic backend as a pattern for provider boundary and dependency discipline, not as a
runtime dependency. Re-check official ElevenLabs docs before implementation because endpoint
contracts and tier restrictions can change.

