# Public API (Integrator Contract)

This document defines the intended, supported API surface for third‑party integration.

## Primary entry point

- `abstractvoice.VoiceManager`

## TTS

- `speak(text: str, speed: float = 1.0, callback=None, voice: str | None = None) -> bool`
  - Plays audio locally.
  - Optional: if `voice` is provided, treat it as a cloned voice_id (requires `abstractvoice[cloning]`).

- `pause_speaking() -> bool`, `resume_speaking() -> bool`, `stop_speaking() -> bool`
  - Playback control.

- `speak_to_bytes(text: str, format: str = "wav", voice: str | None = None) -> bytes`
  - Returns audio bytes for network transmission / headless use.
  - Optional: if `voice` is provided, treat it as a cloned voice_id (requires `abstractvoice[cloning]`).

- `speak_to_file(text: str, output_path: str, format: str | None = None, voice: str | None = None) -> str`
  - Writes an audio file and returns the path.
  - Optional: if `voice` is provided, treat it as a cloned voice_id (requires `abstractvoice[cloning]`).

## STT

- `transcribe_file(audio_path: str, language: str | None = None) -> str`
  - Transcribes audio from a file.

- `transcribe_from_bytes(audio_bytes: bytes, language: str | None = None) -> str`
  - Transcribes audio sent over the network.

## Microphone capture (local assistant mode)

- `listen(on_transcription, on_stop=None) -> bool`
  - Starts microphone capture + VAD + STT in-process.
  - Stop phrase: saying **"ok stop"** (or **"okay stop"**) triggers `on_stop` (if provided).

- `stop_listening() -> bool`
  - Stops microphone capture.

- `pause_listening() -> bool`, `resume_listening() -> bool`
  - Pauses/resumes audio processing while keeping the listening thread alive.

## Voice modes (anti self-interruption)

VoiceManager supports voice modes that control how listening behaves during TTS:

- **wait**: suppresses normal transcriptions and disables “interrupt on speech” during TTS playback (most robust without echo cancellation; stop phrase still works)
- **full**: keeps listening, but disables “interrupt on speech” during TTS to avoid self-interruption

In both modes, the stop phrase remains available as a safe “barge-in” substitute.

## Configuration

- `VoiceManager(..., allow_downloads: bool = True)`
  - Controls whether model weights may be downloaded implicitly.
  - The REPL sets this to `False` (offline-first; no surprise downloads).

- `set_language(language: str) -> bool`
  - For default engines, supported language codes include: `en, fr, de, es, ru, zh`.

- `set_whisper(model_name: str) -> bool`
  - Controls STT model sizing (used for faster‑whisper adapter sizing).

- `enable_aec(enabled: bool = True, stream_delay_ms: int = 0) -> bool`
  - Opt-in acoustic echo cancellation support for true barge-in (requires `abstractvoice[aec]`).
  - Intended for `voice_mode="full"`.

## Callbacks

VoiceManager exposes audio lifecycle hooks:

- `vm.tts_engine.on_playback_start` / `vm.tts_engine.on_playback_end`
- `vm.on_audio_start` / `vm.on_audio_end`
- `vm.on_audio_pause` / `vm.on_audio_resume`

## What is explicitly *not* part of the core contract

- Legacy voice catalogs and legacy model download flows
- CLI convenience behavior

Those can exist, but should not define the library’s integration story.

## Explicit downloads (recommended)For offline deployments, prefetch explicitly:```bash
python -m abstractvoice download --piper en
python -m abstractvoice download --stt small
python -m abstractvoice download --openf5
python -m abstractvoice download --chroma
```