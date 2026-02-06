# API (Integrator contract)

This document defines the intended, supported API surface for third‑party integration.

If you're new, start with `README.md`, then `docs/getting-started.md`, then `docs/overview.md`.
If you're troubleshooting behavior, check `docs/faq.md` and `docs/repl_guide.md`.
If you’re changing internals, `docs/architecture.md` maps the implementation to files.

Code evidence (where these methods live):
- `abstractvoice/voice_manager.py` → `abstractvoice/vm/manager.py` (constructor + wiring)
- `abstractvoice/vm/tts_mixin.py` (TTS + cloning methods)
- `abstractvoice/vm/stt_mixin.py` (STT + listening methods)
- `abstractvoice/vm/core.py` (voice-mode behavior during playback)

## Primary entry point

- `abstractvoice.VoiceManager`

```python
from abstractvoice import VoiceManager

vm = VoiceManager(language="en", allow_downloads=True)
```

## Constructor (most-used knobs)

The source of truth is `abstractvoice/vm/manager.py`:

```python
VoiceManager(
    language: str = "en",
    tts_model: str | None = None,
    whisper_model: str = "base",
    debug_mode: bool = False,
    tts_engine: str = "auto",
    stt_engine: str = "auto",
    allow_downloads: bool = True,
    cloned_tts_streaming: bool = True,
    cloning_engine: str = "f5_tts",
)
```

Notes:
- `allow_downloads` gates *implicit* model downloads in adapters. The REPL sets `False` (offline-first).
- `whisper_model` controls the faster‑whisper model size used by `listen()` and `transcribe_*()`.
- `tts_engine` currently supports only `auto|piper` (Piper-only core).
- `stt_engine` is currently `auto|faster_whisper` for the adapter path.
- `tts_model` is reserved/back-compat (Piper selection is language-driven today).

Supported language codes for the default Piper mapping: `en, fr, de, es, ru, zh` (see `abstractvoice/config/voice_catalog.py` and `abstractvoice/adapters/tts_piper.py`).

## TTS (text → audio)

- `speak(text: str, speed: float = 1.0, callback=None, voice: str | None = None, *, sanitize_syntax: bool = True) -> bool`
  - Plays audio locally (non-blocking playback; synthesis time depends on backend).
  - If `voice` is provided, it is treated as a cloned `voice_id` (requires `abstractvoice[cloning]`).
  - By default, common Markdown syntax is stripped from spoken output (headers + emphasis). Set `sanitize_syntax=False` to speak raw text.

- `set_speed(speed: float) -> bool`, `get_speed() -> float`
  - Adjusts the default speaking speed used by `speak_to_*()` and the REPL.

- `pause_speaking() -> bool`, `resume_speaking() -> bool`, `stop_speaking() -> bool`
  - Playback control.

- `is_speaking() -> bool`, `is_paused() -> bool`
  - Playback state helpers.

- `speak_to_bytes(text: str, format: str = "wav", voice: str | None = None, *, sanitize_syntax: bool = True) -> bytes`
  - Headless/server‑friendly: returns encoded audio bytes.

- `speak_to_file(text: str, output_path: str, format: str | None = None, voice: str | None = None, *, sanitize_syntax: bool = True) -> str`
  - Writes an audio file and returns the path.

### Language & voice selection (Piper path)

- `set_language(language: str) -> bool`
  - Switches the active language and loads the matching Piper voice (best-effort).
  - Validation is based on `abstractvoice/config/voice_catalog.py`.

- `get_language() -> str`, `get_language_name(language_code: str | None = None) -> str`

- `get_supported_languages() -> list[str]`

- `list_available_models(language: str | None = None) -> dict`
  - Lists Piper voices/models for CLI display (see `abstractvoice/vm/tts_mixin.py`).
  - Back-compat alias: `list_voices()`.

- `set_voice(language: str, voice_id: str) -> bool`
  - Backward-compatible method; Piper voice selection is currently best-effort.

## STT (audio → text)

- `transcribe_file(audio_path: str, language: str | None = None) -> str`
  - Transcribes audio from a file.

- `transcribe_from_bytes(audio_bytes: bytes, language: str | None = None) -> str`
  - Transcribes audio sent over the network.

### STT configuration

- `set_whisper(model_name: str) -> None | bool`
  - Updates the faster‑whisper model size used for subsequent operations.

- `get_whisper() -> str`

## Microphone capture (local assistant mode)

- `listen(on_transcription, on_stop=None) -> bool`
  - Starts microphone capture + VAD + STT in-process (`abstractvoice/recognition.py`).
  - Stop phrase(s): `"ok stop"`, `"okay stop"`, and (conservatively) `"stop"`; see `abstractvoice/recognition.py` and `abstractvoice/stop_phrase.py`.

- `stop_listening() -> bool`
  - Stops microphone capture.

- `pause_listening() -> bool`, `resume_listening() -> bool`
  - Pauses/resumes audio processing while keeping the listening thread alive.

- `is_listening() -> bool`
  - Whether the background recognizer thread is running.

- `cleanup() -> bool`
  - Best-effort cleanup for long-lived apps (stop listening, stop speaking, release audio resources).

### Advanced tuning (best-effort)

- `change_vad_aggressiveness(aggressiveness: int) -> bool`
  - For advanced mic/VAD tuning; see `abstractvoice/recognition.py`.

## Voice modes (behavior while speaking)

Voice modes control what the microphone loop does *while TTS is playing*. Set via:

- `set_voice_mode(mode: str) -> bool` where `mode ∈ {"full","wait","stop","ptt"}`

Mode semantics (implemented in `abstractvoice/vm/core.py`):

- **full**: keep listening and allow barge‑in (interrupt TTS on detected speech). Best with AEC or headset; speakers can cause self-interruption (mitigations exist; see echo gating in `abstractvoice/recognition.py`).
- **wait**: pause microphone processing while speaking. No barge‑in and no stop‑phrase detection during TTS. Good for strict turn‑taking.
- **stop**: keep listening, but suppress normal transcriptions during TTS and disable “interrupt on any speech”; a rolling stop‑phrase detector stays active so users can say “ok stop” to cut playback.
- **ptt**: push‑to‑talk profile (thresholds tuned for short utterances). During TTS it behaves like **stop** mode; the integrator controls when to start/stop capture.

The REPL defaults to **mic input off**, and recommends `--voice-mode stop` for hands‑free usage; see `docs/repl_guide.md`.

## Acoustic echo cancellation (optional)

- `enable_aec(enabled: bool = True, stream_delay_ms: int = 0) -> bool`
  - Opt‑in AEC support for true barge‑in (requires `abstractvoice[aec]`).
  - Playback audio chunks are fed to the recognizer via `abstractvoice/vm/core.py` → `VoiceRecognizer.feed_far_end_audio()` in `abstractvoice/recognition.py`.

## Voice cloning (optional; heavy)

Requires `pip install "abstractvoice[cloning]"` (and explicit artifact downloads; see `docs/installation.md`).

Core cloning calls:

- `clone_voice(reference_audio_path: str, name: str | None = None, *, reference_text: str | None = None, engine: str | None = None) -> str`
- `speak(..., voice="<voice_id>")` / `speak_to_bytes(..., voice="<voice_id>")` / `speak_to_file(..., voice="<voice_id>")`
- `list_cloned_voices()`, `get_cloned_voice(voice_id: str) -> dict`

Clone management helpers:

- `set_cloned_voice_reference_text(voice_id: str, reference_text: str) -> bool`
- `rename_cloned_voice(voice_id: str, new_name: str) -> bool`
- `delete_cloned_voice(voice_id: str) -> bool`
- `export_voice(voice_id: str, path: str) -> str`, `import_voice(path: str) -> str`
- `set_cloned_tts_quality(preset: str) -> bool` (`fast|balanced|high`)
- `get_cloning_runtime_info() -> dict`
- `unload_cloning_engines(*, keep_engine: str | None = None) -> int` (best-effort memory relief)
- `unload_piper_voice() -> bool` (best-effort memory relief)

For the user-facing workflow and commands, see `docs/repl_guide.md`.

## Metrics (optional)

- `pop_last_tts_metrics() -> dict | None`
  - Best-effort last-utterance stats used by the REPL verbose mode.

## Callbacks & hooks

- Per-utterance callback: `speak(..., callback=...)` (invoked after playback drains).
- TTS lifecycle callbacks: `vm.tts_engine.on_playback_start` / `vm.tts_engine.on_playback_end` (synthesis/queue lifecycle).
- Audio lifecycle callbacks (actual output): `vm.on_audio_start` / `vm.on_audio_end` / `vm.on_audio_pause` / `vm.on_audio_resume` (wired in `abstractvoice/vm/core.py`).

## Explicit downloads (offline-first)

For offline deployments, prefetch explicitly (cross-platform):

```bash
python -m abstractvoice download --stt small
python -m abstractvoice download --piper en
python -m abstractvoice download --openf5
python -m abstractvoice download --chroma
```

Or use the convenience entrypoint:

```bash
abstractvoice-prefetch --stt small
abstractvoice-prefetch --piper en
```

See also: `docs/installation.md`, `docs/model-management.md`, and `docs/voices-and-licenses.md`.

## Non-contract surface (may change without notice)

- CLI behavior (`abstractvoice/examples/*`)
- Internal adapter details and model catalogs beyond the documented defaults
