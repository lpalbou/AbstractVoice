## Task 013: OpenAI audio backends (TTS + STT + custom voices + realtime)

**Date**: 2026-01-23  
**Last updated**: 2026-04-06  
**Status**: Planned  
**Priority**: P1  

---

## Main goals

- Add **OpenAI-backed** implementations for:
  - **TTS** (Text-to-Speech) via OpenAI **Speech API** (`/v1/audio/speech`)
  - **STT** (Speech-to-Text) via OpenAI **Transcription API** (`/v1/audio/transcriptions`)
  - **Voice cloning / custom voice** via OpenAI **Custom Voices** (`/v1/audio/voice_consents` + `/v1/audio/voices`) when enabled for the org
- Provide a clean path for **voice-agent-grade** integrations:
  - streaming transcription deltas
  - streaming audio output
  - optional OpenAI **Realtime API** sessions (server VAD, noise reduction, interruptions)

## Secondary goals

- Keep AbstractVoice **offline-first by default** for local engines; remote/commercial backends must be **explicit opt-in**.
- Keep integration surfaces small, modular, and testable (adapters + typed results).
- Provide **actionable errors** for missing API key, disabled org features (custom voices), quotas, and timeouts.

---

## Context / problem

This task was originally framed as “ship OpenAI-compatible HTTP endpoints inside AbstractVoice”.

Since then, AbstractVoice has evolved primarily as:
- a **library** (`VoiceManager`, adapters, local-first engines), and
- an **AbstractCore capability plugin** (server endpoints should be owned by AbstractCore/AbstractFramework).

**Revised scope**:
- **AbstractCore / AbstractFramework**: owns OpenAI-compatible *server* endpoints (`/v1/audio/*`).
- **AbstractVoice (this repo)**: owns *backends/adapters* that can call OpenAI APIs (and later other commercial providers) and can be wired into those server endpoints via the capability plugin system.

---

## Constraints

- Avoid forcing network dependencies on the default install.
- Do not imply we provide OpenAI’s **compliance/consent** policy.
  - If OpenAI custom voices are used, we must require explicit consent inputs and store provenance.
- Keep interfaces robust for other commercial providers (Deepgram, ElevenLabs, etc.) so this doesn’t become OpenAI-specific glue.

---

## Research updates (what changed since Jan 2026)

### From our deep research docs

- **Cascaded streaming pipeline remains dominant**: STT → LLM/agent → TTS is still the most reliable architecture for tool-using voice agents, with streaming/pipelining driving “feels instant” UX.
  - See: `docs/research/Agentic Voice Assistant Stack Deep Research.md`
  - See: `docs/research/Voice Assistant Architecture Investigation.md`
- **Commercial differentiators** that we should be able to abstract:
  - streaming STT partials + endpointing/turn detection
  - diarization (speakers + timestamps)
  - streaming TTS (time-to-first-audio)
  - barge-in / interruption support
  - governance/compliance gating around custom voices

### From OpenAI docs (2026-04)

OpenAI now offers multiple ways to build audio apps and voice agents:
- **Speech API**: `POST /v1/audio/speech` (TTS, supports `instructions`, `speed`, `response_format` like `wav|pcm|mp3`)
- **Transcription API**: `POST /v1/audio/transcriptions` (STT, supports streaming text deltas)
- **Translation API**: `POST /v1/audio/translations` (speech → English, `whisper-1`)
- **Custom voices**:
  - `POST /v1/audio/voice_consents` (upload consent recording) → returns `cons_...`
  - `POST /v1/audio/voices` (create voice from consent + sample) → returns `voice_...`
- **Realtime API**: low-latency audio-in/audio-out sessions with server-side VAD and interruption controls.
- **Chat Completions with audio**: native audio input/output via audio-capable chat models (useful when you want tool calling + speech output in one API).

References:
- Audio overview: `https://developers.openai.com/api/docs/guides/audio/`
- Speech API: `https://developers.openai.com/api/docs/api-reference/audio/`
- Transcription API: `https://developers.openai.com/api/docs/api-reference/audio/createTranscription`
- Voice consent: `https://developers.openai.com/api/reference/resources/audio/subresources/voice_consents/methods/create/`
- Create custom voice: `https://developers.openai.com/api/reference/resources/audio/subresources/voices/methods/create/`

---

## Proposed design

### Packaging + configuration

- Add optional extra: `abstractvoice[openai]` (depends on the OpenAI SDK).
- Configuration:
  - `OPENAI_API_KEY` (required to use these adapters)
  - optional base URL / org settings if needed (keep consistent with other integrations).
- Add an explicit “network allowed” gate (do not overload `allow_downloads`, which is local-model semantics).

### Adapters (Phase 1: synchronous)

- **TTS**: `OpenAITTSAdapter` (new file under `abstractvoice/adapters/tts_openai.py`)
  - Implements `TTSAdapter`.
  - Uses `POST /v1/audio/speech`.
  - Parameters to expose (adapter params):
    - `model` (default: `gpt-4o-mini-tts`)
    - `voice` (built-in voice name OR custom voice id)
    - `instructions` (tone/acting/style)
    - `speed`
    - `response_format` (`wav` default in AbstractVoice to align with our pipeline)
- **STT**: `OpenAISTTAdapter` (new file under `abstractvoice/adapters/stt_openai.py`)
  - Implements `STTAdapter`.
  - Uses `POST /v1/audio/transcriptions` for transcription; optionally `POST /v1/audio/translations` where needed.
  - Support model selection:
    - low-latency: `gpt-4o-mini-transcribe`
    - higher quality: `gpt-4o-transcribe`
    - diarization (non-latency-sensitive): `gpt-4o-transcribe-diarize`

### Voice cloning / custom voices (Phase 2)

Add a new cloning engine (e.g. `OpenAICustomVoiceCloningEngine`) that maps AbstractVoice “create a voice” to OpenAI’s two-step flow:

1. Upload consent recording: `POST /v1/audio/voice_consents` → `cons_...`
2. Create voice: `POST /v1/audio/voices` with:
   - `name`
   - `audio_sample` (voice reference)
   - `consent` (the `cons_...` id)

Then store the resulting `voice_...` id in the AbstractVoice voice store so it can be reused across sessions/machines (subject to the OpenAI org/account).

Key constraints:
- Custom voices are gated by OpenAI org enablement; handle `404/403` with explicit guidance.
- Do not pretend to provide consent policy: require explicit consent recording inputs, store provenance, and surface “customer responsibility” clearly.

### Streaming primitives (Phase 3+; driven by research + OpenAI capabilities)

AbstractVoice’s current `STTAdapter` returns a single `str`, which is insufficient for:
- streaming deltas (`transcript.text.delta`)
- diarization results (speaker labels + timestamps)
- confidence/logprobs

Proposed approach:
- Keep the existing simple API for compatibility.
- Add typed “rich transcription” results:
  - `TranscriptionResult` (text + optional segments + speaker labels + timestamps + logprobs)
- Add an optional streaming method for adapters that support it:
  - `transcribe_stream(...) -> Iterator[TranscriptEvent]`
- Add an optional streaming method for TTS where supported:
  - `synthesize_stream(...) -> Iterator[AudioChunk]` (prefer PCM16 chunks for low-latency playback)

### Realtime sessions (Phase 4; optional)

Add an “advanced” module that can open a Realtime session and expose it as an event stream:
- audio in (PCM frames)
- transcript deltas out
- audio deltas out
- server-side VAD and interruption controls

---

## Decision

**Chosen approach**: implement OpenAI support as **optional backends** (adapters + cloning engine + streaming primitives), and keep “OpenAI-compatible HTTP endpoints” in **AbstractCore/AbstractFramework**.

**Why**:
- Avoid duplicating two HTTP stacks (Flask inside AbstractVoice vs FastAPI in AbstractCore).
- Keeps AbstractVoice focused on reusable building blocks (adapters + types).
- Aligns with 2026 best practice: **streaming cascade** STT → LLM → TTS, plus optional Realtime sessions.

---

## Implementation plan (phased)

### Phase 1 — Synchronous TTS + STT adapters (MVP)

- **Packaging**
  - Add `abstractvoice[openai]` optional extra.
  - Add a minimal configuration story (`OPENAI_API_KEY`, timeouts).
- **TTS adapter**
  - Implement `OpenAITTSAdapter` mapping to `/v1/audio/speech`.
  - Support `instructions`, `speed`, `response_format` (default `wav`).
  - Map `set_language(...)` best-effort (OpenAI TTS is not strictly locale-gated; language can be expressed in `instructions`).
- **STT adapter**
  - Implement `OpenAISTTAdapter` mapping to `/v1/audio/transcriptions`.
  - Support `language`, `prompt`, `response_format`, and model selection.
- **Wiring**
  - Register `openai` as `tts_engine` and `stt_engine` (adapter registries + `VoiceManager`).
- **Docs**
  - Update `docs/installation.md` (new extra + required env var).
  - Update `docs/api.md` (new engines + supported params).
- **Tests**
  - Unit tests should stub HTTP/SDK calls; no live API calls in CI.

### Phase 2 — Custom voices (voice cloning / “brand voice”)

- Implement `OpenAICustomVoiceCloningEngine`:
  - `create_consent(...)` → `/v1/audio/voice_consents`
  - `create_voice(...)` → `/v1/audio/voices`
- Integrate with AbstractVoice voice store:
  - persist `voice_id` + metadata + provenance
  - allow selection via `tts_voice clone <id>` path when `tts_engine=openai`
- Add UX affordances (optional, but high impact):
  - REPL helpers to create consent/voice, list voices, and select a voice id
  - clear error messages when the org is not eligible/enabled (often `404`)

### Phase 3 — Streaming STT + streaming TTS (voice-agent primitives)

- Add typed outputs:
  - `TranscriptionResult` (segments, speakers, timestamps, logprobs — optional)
  - `TranscriptEvent` (delta/segment/done)
  - `AudioChunk` (PCM bytes + sample rate)
- Add optional streaming methods (do not break existing adapters):
  - `transcribe_stream(...)` for adapters that can stream
  - `synthesize_stream(...)` for adapters that can stream audio output
- Update the faster-whisper adapter to optionally emit segments to validate the abstraction.

### Phase 4 — Realtime API session helper (optional)

- Create `OpenAIRealtimeSession` (standalone module):
  - session creation + configuration
  - push PCM frames in
  - receive transcript deltas + audio deltas out
  - server VAD / interruption controls surfaced as events
- Keep transport concerns out of AbstractVoice:
  - AbstractVoice exposes an event stream API; WebRTC/WebSocket wiring lives in higher layers (e.g. AbstractAssistant / gateway).

---

## Testing strategy

- **Unit tests**: mock the OpenAI SDK client methods and validate parameter mapping + error handling.
- **Optional integration tests** (skipped by default):
  - run only when `OPENAI_API_KEY` is present
  - record that custom voices require org enablement and may not work everywhere

---

## Success criteria

- **TTS**
  - `VoiceManager(tts_engine="openai").speak_to_bytes(..., format="wav")` returns a valid WAV (or a clear error when misconfigured).
- **STT**
  - `VoiceManager(stt_engine="openai").transcribe(path)` returns text.
  - When model is diarization-capable, we can optionally return a rich structure without breaking the `str` API.
- **Custom voices**
  - With org enablement: consent + voice creation yields a reusable `voice_id` persisted in the local store.
  - Without enablement: a clear, actionable error message explains the constraint.
- **Architecture**
  - No “server stack” is added to AbstractVoice as a hard dependency.
  - AbstractCore/AbstractFramework can use these backends via the plugin path.

---

## Separate backlog: AWS backends (not in this task)

AWS audio services (Polly, Transcribe, and Bedrock voice-agent capabilities) should be tracked separately so this task can stay focused on OpenAI first.

Planned follow-up (to be created): `docs/backlog/planned/032_aws_audio_backends.md`

