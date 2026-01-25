## Task 025: Make tests headless/CI-safe (no audio devices required)

**Date**: 2026-01-23  
**Status**: Planned  
**Priority**: P0  

---

## Main goals

- Make `pytest -q` pass in CI/headless environments **without** requiring real audio input/output devices.
- Eliminate background-thread exceptions during the test suite.

## Secondary goals

- Clearly separate **unit tests** (default) from **hardware/integration tests** (opt-in).

---

## Context / problem

Current tests assume working PortAudio devices:

- `tests/test_callbacks.py` calls `VoiceManager.speak()` which opens a `sounddevice.OutputStream`. In headless environments (or runners without a default audio output device), this fails with `sounddevice.PortAudioError: Error querying device -1`.
- `tests/test_fresh_install.py::test_cli_commands` instantiates `VoiceREPL()`, which currently **auto-enables voice input** in `__init__` and attempts to open `sounddevice.InputStream`. This produces `PytestUnhandledThreadExceptionWarning` when no microphone is available.

This makes the suite brittle and prevents reliable CI.

---

## Constraints

- Keep `VoiceManager.speak/listen/transcribe/pause/resume/stop/...` stable.
- Avoid “test-only” behavior in production code; if new switches are added, they must be generally useful (e.g., a headless/server mode).
- Keep dependencies minimal; do not add heavy test frameworks unless necessary.

---

## Research, options, and references

- **Option A: Skip device-dependent tests when no devices exist**
  - Use `pytest.skip()` (or markers) if `sounddevice.query_devices()` fails or no default devices exist.
  - Trade-offs: avoids failures but reduces coverage of callback/timing logic.
  - References:
    - `https://docs.pytest.org/en/stable/how-to/skipping.html`
    - `https://python-sounddevice.readthedocs.io/` (device querying / PortAudio behavior)

- **Option B: Mock the audio backend for deterministic tests**
  - Provide a fake `OutputStream`/`InputStream` (or an injectable backend) so callback ordering/state can be tested without hardware.
  - Trade-offs: small refactor or careful monkeypatching; highest stability and coverage.
  - References:
    - `https://docs.pytest.org/en/stable/how-to/monkeypatch.html`

- **Option C: Add an explicit “headless” mode for playback/capture**
  - E.g., environment variable or constructor option to disable real audio devices and use null backends.
  - Trade-offs: adds a supported runtime mode (useful for servers) but requires clear semantics for `speak()`/`listen()`.

---

## Decision

**Chosen approach**: combine **Option B** (default unit tests) with **Option A** for true hardware tests.

**Why**:
- Keeps `pytest -q` stable everywhere while preserving meaningful coverage of callback ordering/state.
- Allows optional “real device” tests to remain available for local QA.

---

## Dependencies

- **ADRs**:
  - `docs/adr/0002_barge_in_interruption.md` (STT/TTS coordination and audio pipeline implications)
- **Backlog tasks**: none

---

## Implementation plan

- Convert `tests/test_callbacks.py` into a deterministic test by:
  - monkeypatching `sounddevice.OutputStream` (or `_import_sounddevice()` in `abstractvoice/tts/tts_engine.py`) with a fake stream that exercises the callback path, or
  - introducing a small injectable audio-backend seam in `NonBlockingAudioPlayer` (preferred if it also benefits integrators).
- Ensure the REPL does not auto-start microphone capture during tests:
  - Add a `VoiceREPL(..., auto_voice: bool = True)` flag (or equivalent), defaulting to current behavior for humans but disabled in tests, OR
  - Gate auto-start behind `sys.stdin.isatty()` and/or an env var like `ABSTRACTVOICE_NO_AUTO_VOICE=1`.
- Add pytest markers:
  - `integration` / `requires_audio_device` for tests that truly require hardware.
  - Default `pytest -q` runs only unit tests.
- Ensure any voice-recognition threads started during tests are shut down and do not emit unhandled exceptions.

---

## Success criteria

- `pytest -q` passes on runners with **no** audio devices.
- Test output has **no** `PytestUnhandledThreadExceptionWarning`.
- Hardware-dependent tests can still be run explicitly (e.g., `pytest -m requires_audio_device`).

---

## Test plan

- `pytest -q`
- Optional local hardware run:
  - `pytest -m requires_audio_device -q`

