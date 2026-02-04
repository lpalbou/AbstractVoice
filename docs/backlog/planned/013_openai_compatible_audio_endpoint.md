## Task 013: OpenAI-compatible audio endpoints (speech + STT + streaming)

**Date**: 2026-01-23  
**Status**: Planned  
**Priority**: P2 (optional/demo)  

---

## Main goals

- Provide HTTP endpoints compatible with the **shape + semantics** of OpenAI Audio endpoints so third-party tooling can swap to AbstractVoice with minimal glue:
  - `POST /v1/audio/speech`
  - `POST /v1/audio/transcriptions`
  - `POST /v1/audio/translations`
- Provide a **streaming audio strategy** for long texts (OpenAI-style streaming events + robust chunking) to avoid long blocking synthesis.

## Secondary goals

- Keep the server implementation small and dependency-light (no system deps; headless-friendly).
- Keep the core `VoiceManager` integrator contract unchanged.
- Make streaming **invisible for end users** (simple API), but **clearly documented** for third-party integrators.

---

## Context / problem

AbstractVoice will ultimately be integrated into AbstractFramework (which already serves OpenAI-compatible endpoints). However, having a minimal OpenAI-shaped endpoint inside AbstractVoice is useful for:

- demos of the voice modality
- compatibility checks
- quick third-party integrations that expect `/v1/audio/speech`

Coordination note (2026-01-29):
- AbstractCore Server now ships OpenAI-compatible **vision generation** endpoints (`/v1/images/*`) by delegating to `abstractvision`.
- We should decide whether OpenAI-compatible **audio** endpoints live:
  - **inside AbstractVoice** (this task; Flask, minimal dependency footprint), or
  - **inside AbstractCore Server** as an optional `audio_endpoints.py` router (FastAPI, keeps the “OpenAI-compatible gateway” surface unified).
This task keeps the original “inside AbstractVoice” plan, but AbstractFramework integration work should reassess placement before implementation to avoid duplicating two HTTP stacks.

---

## Constraints

- Piper-only core TTS (no legacy Coqui).
- Cross-platform install (Windows/macOS/Linux).
- Clear errors for unsupported formats/voices.
- Don’t implement legal/compliance policy: if OpenAI has endpoints for voice consent and custom voice creation, we can document mapping and constraints, but must not pretend to provide equivalent compliance.

---

## Research, options, and references

- **OpenAI Audio API references (Jan 2026)**:
  - `POST /v1/audio/speech`: `https://platform.openai.com/docs/api-reference/audio/createSpeech`
  - `POST /v1/audio/voices` (custom voices): `https://platform.openai.com/docs/api-reference/audio/createVoice`
  - `POST /v1/audio/transcriptions`: `https://platform.openai.com/docs/api-reference/audio/createTranscription`
  - `POST /v1/audio/translations`: `https://platform.openai.com/docs/api-reference/audio/createTranslation`
  - Voice object: `https://platform.openai.com/docs/api-reference/audio/voice-object`
  - Speech streaming events:
    - `speech.audio.delta`: `https://platform.openai.com/docs/api-reference/audio/speech-audio-delta-event`
    - `speech.audio.done`: `https://platform.openai.com/docs/api-reference/audio/speech-audio-done-event`
  - Transcript streaming events:
    - `transcript.text.delta`: `https://platform.openai.com/docs/api-reference/audio/transcript-text-delta-event`
    - `transcript.text.segment`: `https://platform.openai.com/docs/api-reference/audio/transcript-text-segment-event`
    - `transcript.text.done`: `https://platform.openai.com/docs/api-reference/audio/transcript-text-done-event`
  - Transcription JSON shapes:
    - `json`: `https://platform.openai.com/docs/api-reference/audio/json-object`
    - `diarized_json`: `https://platform.openai.com/docs/api-reference/audio/diarized-json-object`
    - `verbose_json`: `https://platform.openai.com/docs/api-reference/audio/verbose-json-object`

- **Streaming + chunking best practices (industry + research)**:
  - Chunk by **sentence/clause boundaries** for naturalness and better latency/quality tradeoffs (common guidance):  
    `https://developers.deepgram.com/docs/tts-text-chunking`
  - Latency optimization (streaming recommended):  
    `https://elevenlabs.io/docs/best-practices/latency-optimization`
  - Research direction (“speak while you think” / interleaved streaming):  
    `https://arxiv.org/abs/2309.11210`, `https://arxiv.org/html/2505.19206v1`
- **Framework choice**:
  - Option A: Flask (already an optional dependency group in `pyproject.toml`)
  - Option B: FastAPI (would add a new dependency stack)
  - Decision: prefer **Flask** for minimal change.

---

## Decision

Implement Phase 1 as a **Flask** server (keeps dependency footprint small; already present as optional extra) with a clearly scoped compatibility layer.

### Endpoints we will implement in AbstractVoice (Phase 1)

- `POST /v1/audio/speech`
  - Accept JSON body compatible with OpenAI’s “create speech” (at minimum `model`, `input`, `voice`).
  - Output:
    - Non-streaming: returns full audio bytes.
    - Streaming: supports OpenAI-style streaming events (`speech.audio.delta`/`speech.audio.done`) via SSE.
  - Voice mapping:
    - Built-in voices map to our **Piper language/voice selection**.
    - Custom voice objects `{ "id": "..." }` map to our **cloned voice_id** (optional feature).

- `POST /v1/audio/transcriptions`
  - Accept multipart form (`file`, `model`, optional `language`, `response_format`, `stream`).
  - Return:
    - `text` for `response_format=text`
    - JSON objects for `json` / `verbose_json` (and optionally `diarized_json` if diarization is implemented later).

- `POST /v1/audio/translations`
  - Same as transcriptions, but constrained to returning English (Phase 1 can implement as “transcribe then translate” if/when translation is supported; otherwise return 400 with a clear message).

### Endpoints we will explicitly NOT implement in AbstractVoice (Phase 1)

- `POST /v1/audio/voices` and voice consent endpoints (OpenAI includes consent workflows; we cannot provide equivalent compliance policy in a small OSS library).
  - We can *document* a mapping:
    - our `clone_voice(...)` / store ↔ “custom voice”
  - But we should not ship an endpoint that implies legal consent management parity.
  - If desired, implement these endpoints at the AbstractFramework layer where policy can live.

---

## Dependencies

- ADRs:
  - `docs/adr/0001-local_assistant_out_of_box.md` (no system deps)
- Backlog:
  - Uses Piper defaults established in earlier tasks

---

## Implementation plan

### Server structure

- Replace/extend `abstractvoice/examples/web_api.py` with:
  - `create_app(...)` factory
  - modular blueprints:
    - `speech_routes.py`
    - `stt_routes.py`
- Keep it dependency-injected so `VoiceManager` can be swapped/mocked in tests.

### Streaming/chunking strategy (must be robust + simple)

Goal: reduce “time to first audio” and avoid huge single-shot synthesis for long text.

- **Text chunking**:
  - Prefer sentence/clause boundaries; fall back to max-char slicing.
  - Avoid obvious pitfalls (empty chunks, runaway chunk size).
  - A deterministic chunker is easier to reason about and test.
- **Audio streaming**:
  - For each text chunk:
    - synthesize chunk audio (Piper or cloning engine)
    - immediately emit `speech.audio.delta` (base64 audio)
  - End with `speech.audio.done`.
  - This provides incremental audio output even if the underlying engine is not “true streaming”.

### Response formats

- Support OpenAI response formats when feasible:
  - speech: `wav` + possibly `pcm` later (Phase 2)
  - transcription:
    - `text` (Phase 1)
    - `json` (Phase 1)
    - `verbose_json` (Phase 1, best-effort)
    - `diarized_json` (Phase 2, only when diarization exists)

---

## Success criteria- `POST /v1/audio/speech` returns WAV bytes (`RIFF...WAVE`) for valid non-stream requests.
- Streaming speech returns a valid SSE stream with `speech.audio.delta` chunks and a final `speech.audio.done`.
- `POST /v1/audio/transcriptions` returns valid outputs for `response_format=text|json`.
- Unsupported `format` returns a clear 400 error.
- Tests pass: `pytest -q`.