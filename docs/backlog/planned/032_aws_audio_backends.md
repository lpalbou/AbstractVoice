## Task 032: AWS audio backends (Polly TTS + Transcribe STT)

**Date**: 2026-04-06  
**Status**: Planned  
**Priority**: P3  

---

## Main goals

- Add optional AWS-backed adapters for:
  - **TTS** via **Amazon Polly** (speech synthesis)
  - **STT** via **Amazon Transcribe** (batch + streaming transcription)
- Keep the integration compatible with:
  - AbstractVoice’s adapter contracts (`TTSAdapter`, `STTAdapter`)
  - AbstractCore capability plugins (so AbstractCore’s `/v1/audio/*` endpoints can route to AWS)

## Secondary goals

- Support **streaming** where AWS provides it (especially Transcribe Streaming).
- Provide an explicit and safe **credentials/config story** (no implicit credential leakage; clear errors).

---

## Context / problem

Our deep research notes highlight AWS as a strong enterprise option for voice systems:
- Amazon Transcribe Streaming (bidirectional HTTP/2 / WebSocket) for low-latency STT
- Amazon Polly (Standard/Neural/Generative) for reliable multilingual TTS, plus “Brand Voice” for custom voices

See:
- `docs/research/Agentic Voice Assistant Stack Deep Research.md`
- `docs/research/Voice Assistant Architecture Investigation.md`

This is intentionally a **separate backlog item** from OpenAI integration (Task 013) so we can ship OpenAI first.

---

## Constraints

- Do not add AWS dependencies to the default install.
- Prefer the canonical AWS SDK (Python: `boto3`), but keep the adapter layer thin.
- Do not attempt to implement enterprise “Brand Voice” onboarding/compliance; surface those as gated programs.

---

## Research, options, and references

- Amazon Polly
  - Docs: `https://docs.aws.amazon.com/polly/`
  - Synthesis is returned as a stream in many SDKs (good fit for a future `synthesize_stream(...)`).
- Amazon Transcribe
  - Docs: `https://docs.aws.amazon.com/transcribe/`
  - Streaming STT options include bidirectional streaming interfaces; feature constraints exist around language ID + other options.

---

## Decision

**Chosen approach**: implement AWS support as optional adapters (`abstractvoice[aws]`) with a clear config surface, and extend streaming primitives in AbstractVoice (if needed) rather than hard-coding provider-specific streaming into the core APIs.

---

## Implementation plan (high-level)

### Phase 1 — Batch adapters (lowest risk)

- Add optional extra: `abstractvoice[aws]` (depends on `boto3`)
- Implement:
  - `AwsPollyTTSAdapter` (maps to Polly synthesize speech)
  - `AwsTranscribeSTTAdapter` (maps to Transcribe batch transcription jobs)
- Document credentials/config:
  - region, credential provider chain, explicit env vars
- Provide stubbed unit tests (mock AWS SDK client calls).

### Phase 2 — Streaming STT (voice-agent relevance)

- Implement `transcribe_stream(...)` via Transcribe Streaming.
- Map streaming events into AbstractVoice’s typed `TranscriptEvent` abstraction (once that exists).

### Phase 3 — Streaming TTS (optional)

- If Polly synthesis can be consumed incrementally, expose `synthesize_stream(...)` for low time-to-first-audio playback.

---

## Success criteria

- A user can select `tts_engine="aws_polly"` and `stt_engine="aws_transcribe"` and get correct results with clear errors when misconfigured.
- Streaming STT works end-to-end in an example pipeline (optional Phase 2).

