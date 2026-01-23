## Task 009: Voice cloning (candidate + adapter implementation, permissive)

**Date**: 2026-01-23  
**Status**: Completed  
**Priority**: P0  

---

## Summary

Implemented an optional, permissive-licensed voice cloning pathway behind `abstractvoice[cloning]`, using:

- **F5-TTS** code (MIT) + **OpenF5-TTS-Base** weights (Apache 2.0)
- Local reference audio bundling + portable import/export
- Auto-generation of `ref_text` via built-in STT (faster-whisper) when not provided

This keeps AbstractVoice core clean (Piper-first), while enabling cloning as an opt-in extra.

---

## Changes

- Added `abstractvoice/cloning/`:
  - `VoiceCloneStore`: stores reference bundles + metadata; supports zip export/import
  - `VoiceCloner`: high-level manager
  - `F5TTSVoiceCloningEngine`: wraps `f5-tts_infer-cli`, downloads OpenF5 artifacts, prepares reference WAV(s)

- Integrated into `VoiceManager` (without breaking existing contract):
  - `clone_voice()`, `list_cloned_voices()`, `export_voice()`, `import_voice()`
  - `speak_to_bytes(..., voice=<voice_id>)` routes to the cloning engine when `voice` is set

- Added optional deps:
  - `pyproject.toml` extra: `cloning = ["f5-tts>=1.1.0"]`

---

## Validation

- Unit tests added for the store: `tests/test_voice_cloning_store.py`
- Test suite: **30 passed, 1 skipped**

---

## Manual smoke test (HAL9000 samples)

Use WAV references from `audio_samples/hal9000/` (MP3 is intentionally not supported without extra system deps).

Example workflow:

- `voice_id = vm.clone_voice("audio_samples/hal9000/hal9000_hello.wav", name="hal9000")`
- `wav = vm.speak_to_bytes("Good evening, Dave.", voice=voice_id, format="wav")`
- Verify WAV header: `wav[:4] == b"RIFF"`

---

## References

- F5-TTS MIT license: `https://github.com/SWivid/F5-TTS/blob/main/LICENSE`
- OpenF5-TTS-Base (Apache 2.0 weights): `https://huggingface.co/mrfakename/OpenF5-TTS-Base`
