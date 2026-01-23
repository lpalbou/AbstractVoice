## Task 003: VoiceManager Adapter Integration Hardening

**Status**: Planned  
**Priority**: P0 (Blocks “all tests green” / stability)  
**Scope**: Internal integration (no public API breaking changes)

---

## Problem Statement (What we observed)

We introduced adapter-based engines (Piper TTS + faster-whisper STT) but `VoiceManager` still had implicit assumptions from the legacy stack:

- `VoiceManager.tts_engine` is always a TTSEngine-like object (stop/pause/resume callbacks, `audio_player`, etc.)
- `VoiceManager.transcribe_file()` uses the default STT path without requiring optional `openai-whisper`

This mismatch caused functional crashes (e.g. `NoneType.stop()` / missing callbacks) and made “Tasks 001/002 complete” true only in isolation, not in the full system.

Additionally, we observed **intermittent interpreter crashes at process exit** during test runs involving audio playback and/or rapid engine switching (a PortAudio/cffi callback lifecycle issue). This is not a test-specific special case; it’s a real stability risk for user programs that start/stop audio frequently.

---

## Success Criteria

- **Functional correctness**:
  - `VoiceManager` works with default engines (`piper` + `faster-whisper`) without optional deps.
  - Backward-compat calls (`stop_speaking`, `pause_speaking`, `resume_speaking`, callback wiring) do not crash under Piper.
  - `transcribe_file()` works with faster-whisper by default; legacy `openai-whisper` remains optional.

- **Stability**:
  - `pytest -q` exits cleanly (no segfault / abort) on macOS.
  - Audio streams are reliably closed on `cleanup()` and at interpreter exit.

---

## Proposed Design (Robust, general-purpose)

### 1) Enforce a Stable Internal Contract (Facade)
Create a TTSEngine-compatible facade for adapter-based TTS:
- Wrap `TTSAdapter` behind an engine interface with `speak/stop/pause/resume/is_active/is_paused`
- Use the existing `NonBlockingAudioPlayer` (single responsibility: playback control)
- Provide the same callback surface (`on_playback_start/on_playback_end` + audio lifecycle)

**Why**: `VoiceManager` must coordinate voice modes + interruptions reliably regardless of which synthesis backend is active.

### 2) Make STT Default Path Use faster-whisper
Update `VoiceManager.transcribe_file()` / `transcribe_from_bytes()` to prefer the STT adapter (`FasterWhisperAdapter`) when available.

**Why**: Base install should not require legacy `openai-whisper`.

### 3) Hardening Against Audio Shutdown Crashes
Add defensive cleanup:
- Always stop/abort PortAudio streams during `VoiceManager.cleanup()`
- Track live players and stop them at process exit (`atexit`)
- Ensure adapter-engine stop/cleanup works even when playback finished “just now”

**Why**: stability must hold in real-world usage (rapid start/stop, short clips, CLI usage).

### 4) Testing Strategy Adjustment (Without “special-casing”)
Keep tests real, but avoid nondeterministic process-exit behavior:
- Treat “fresh install” smoke scenarios as **separate-process** tests (spawn a new Python process per scenario) so that any low-level audio teardown doesn’t poison the whole test runner.

**Why**: this mirrors real usage better and makes the suite reliable.

---

## Implementation Steps

- Add adapter TTSEngine facade in `abstractvoice/tts/`
- Wire it in `VoiceManager` whenever a TTS adapter is selected
- Add STT adapter selection and route `transcribe_file()` to it
- Add robust shutdown hooks and ensure `cleanup()` releases audio resources
- Update/extend tests to validate:
  - Callback wiring works under Piper
  - Language/voice switching doesn’t crash under Piper
  - Test runner exits cleanly

