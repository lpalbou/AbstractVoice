## Task 013: OpenAI-compatible audio endpoint (speech synthesis + transcription)

**Status**: Planned  
**Priority**: P2 (optional/demo)  

---

## Goal

Provide an HTTP API that is intentionally compatible (shape + semantics) with:

- OpenAI `POST /v1/audio/speech` (createSpeech): `https://platform.openai.com/docs/api-reference/audio/createSpeech`

So that third-party tooling expecting an OpenAI-style endpoint can switch to AbstractVoice with minimal glue.

**Context**: AbstractVoice will ultimately be integrated into AbstractFramework, which already serves OpenAI-compatible endpoints for broader functionality. This task is optional and primarily intended for:
- quick demos of voice modality
- integration tests / compatibility checks
- third-party tools that expect `/v1/audio/speech`

---

## Scope (Phase 1)

### 1) `POST /v1/audio/speech`

Request fields to support:
- `model`: string (map to internal TTS engine selection; default Piper)
- `input`: string (text)
- `voice`: string (map to Piper voice/language OR legacy voice ids; document mapping)
- `format`: `wav|mp3|ogg` (match OpenAI accepted values where possible)
- optional: `speed` (if supported)

Response:
- audio bytes stream with correct `Content-Type`

### 2) (Optional) `POST /v1/audio/transcriptions`

If we want parity on STT as well (faster-whisper default).

---

## Design constraints

- No extra system dependencies (matches ADR 0001).
- Headless-friendly (server use case).
- Clear error messages when a requested voice/model is not available.

---

## Proposed implementation

- Add a small server module (likely Flask or FastAPI; decide based on existing deps and simplicity).
- Use `VoiceManager.speak_to_bytes()` for synthesis (Piper path).
- Use `VoiceManager.transcribe_from_bytes()` for STT endpoint (if included).
- Provide a documented mapping table: OpenAI `voice` → AbstractVoice voice/language.

