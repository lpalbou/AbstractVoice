## Task 006: REPL as a First-Class “Voice Assistant” Product

**Status**: ✅ Completed  
**Completed**: 2026-01-23  
**Priority**: P1

---

## What was delivered

The REPL (`python -m abstractvoice cli`) now exposes explicit commands mapping to the integrator API:

- `/speak <text>`: TTS without calling the LLM
- `/transcribe <path>`: STT from file via `VoiceManager.transcribe_file()` (faster-whisper default)
- `/tts_engine <auto|piper>`: select TTS engine
- `/stt_engine <auto|faster_whisper|whisper>`: select STT engine

And it remains usable in text-only mode without crashing.

---

## Documentation updates

- Updated `docs/repl_guide.md` with an explicit smoke-test checklist.

---

## Verification

- Full test suite: **26 passed, 1 skipped**
