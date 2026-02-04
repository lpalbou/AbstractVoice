# AbstractVoice architecture

This document describes how AbstractVoice works internally (v`0.6.0`), and where to look in the code when you need to change behavior.

If you want the supported integrator contract, start with `docs/public_api.md`. For REPL behavior and commands, see `docs/repl_guide.md`.

For acronyms used here (TTS/STT/VAD/VM/MM), see `docs/acronyms.md`.

## TL;DR

- `abstractvoice.VoiceManager` is the orchestration façade (`abstractvoice/vm/*`).
- **TTS (default)**: Piper adapter → `AdapterTTSEngine` → `NonBlockingAudioPlayer` (pause/resume/stop).
- **STT (default)**: `VoiceRecognizer` loop (mic capture) → `VoiceDetector` (webrtcvad) → `FasterWhisperAdapter`.
- Voice modes are implemented by wiring TTS playback callbacks to recognizer controls (`abstractvoice/vm/core.py`).

## Component diagram

```mermaid
flowchart LR
  App[Your app / REPL] <--> VM[VoiceManager]

  VM -->|speak()*| TTSEngine[AdapterTTSEngine]
  TTSEngine -->|synthesize| Piper[PiperTTSAdapter]
  TTSEngine --> Player[NonBlockingAudioPlayer]
  Player --> Out[(sounddevice OutputStream)]

  VM -->|listen()*| Rec[VoiceRecognizer]
  Rec --> In[(sounddevice InputStream)]
  Rec --> VAD[VoiceDetector (webrtcvad)]
  Rec --> STT[FasterWhisperAdapter]

  Player -. on_audio_chunk (optional AEC) .-> Rec
```

## Code map (evidence)

Start points (in call order):

- Public façade: `abstractvoice/voice_manager.py` → `abstractvoice/vm/manager.py`
- TTS orchestration: `abstractvoice/vm/tts_mixin.py`
- STT/listening orchestration: `abstractvoice/vm/stt_mixin.py`
- Playback/lifecycle wiring + voice modes: `abstractvoice/vm/core.py`

TTS implementation:

- Piper adapter: `abstractvoice/adapters/tts_piper.py`
- TTS engine wrapper (back-compat contract): `abstractvoice/tts/adapter_tts_engine.py`
- Low-latency audio player: `abstractvoice/tts/tts_engine.py`

STT implementation:

- Mic/VAD/STT loop: `abstractvoice/recognition.py`
- VAD wrapper: `abstractvoice/vad/voice_detector.py`
- faster-whisper adapter: `abstractvoice/adapters/stt_faster_whisper.py`
- Stop phrase normalization: `abstractvoice/stop_phrase.py`

Optional features:

- AEC (extra): `abstractvoice/aec/webrtc_apm.py` (used by `abstractvoice/recognition.py`)
- Voice cloning (extra): `abstractvoice/cloning/*` (used by `abstractvoice/vm/tts_mixin.py`)
- AbstractCore plugin: `abstractvoice/integrations/abstractcore_plugin.py`

## Data flows

### TTS (local playback)

1) Your app calls `VoiceManager.speak()` (`abstractvoice/vm/tts_mixin.py`).
2) Piper path (default): `AdapterTTSEngine.speak()` (`abstractvoice/tts/adapter_tts_engine.py`)
   - synthesizes audio via `PiperTTSAdapter.synthesize()` (`abstractvoice/adapters/tts_piper.py`)
   - enqueues audio into `NonBlockingAudioPlayer.play_audio()` (`abstractvoice/tts/tts_engine.py`)
3) Playback runs in the PortAudio callback thread (`sounddevice.OutputStream`).

Pause/resume is implemented by toggling a lock-protected paused flag inside the audio callback (see `NonBlockingAudioPlayer.pause()/resume()`).

### STT (microphone listening)

1) Your app calls `VoiceManager.listen()` (`abstractvoice/vm/stt_mixin.py`).
2) A `VoiceRecognizer` instance is created (`abstractvoice/recognition.py`) with:
   - a VAD (`VoiceDetector`, `abstractvoice/vad/voice_detector.py`)
   - an STT adapter (`FasterWhisperAdapter`, `abstractvoice/adapters/stt_faster_whisper.py`)
3) The recognizer thread opens a `sounddevice.InputStream` and loops:
   - optional AEC preprocessing
   - VAD detection and buffering
   - transcription and callback emission

Stop phrase behavior:
- The recognizer checks stop phrases on completed transcriptions, and can also run a low-rate rolling detector while normal transcriptions are suppressed (see `_maybe_detect_stop_phrase_continuous()` in `abstractvoice/recognition.py`).

## Coordination: voice modes while speaking

AbstractVoice wires TTS lifecycle to listening behavior in `abstractvoice/vm/core.py`:

- `tts_engine.on_playback_start` → `VoiceManagerCore._on_tts_start()`
- `tts_engine.on_playback_end` → `VoiceManagerCore._on_tts_end()`

`set_voice_mode(mode)` is public (`abstractvoice/vm/stt_mixin.py`). Modes:

- **full**: keep listening; allow barge‑in (interrupt TTS on detected speech). Intended for AEC/headset; speakers may self-interrupt (mitigated by echo gating heuristics in `abstractvoice/recognition.py`).
- **wait**: pause mic processing while speaking (`VoiceRecognizer.pause_listening()` / `resume_listening()`).
- **stop**: keep mic processing, but suppress normal transcriptions while speaking and disable speech-triggered interruption; a stop-phrase detector remains active (`pause_transcriptions()` + rolling stop detector).
- **ptt**: push‑to‑talk profile (thresholds tuned for short utterances). While speaking it behaves like **stop** mode; capture is controlled by the integrator/REPL.

Design decisions behind these modes:
- ADR 0001: `docs/adr/0001-local_assistant_out_of_box.md`
- ADR 0002: `docs/adr/0002_barge_in_interruption.md`

## Threading model (practical)

- **Main thread**: your app / REPL.
- **Recognizer thread**: mic capture + VAD + STT (`VoiceRecognizer._recognition_loop()`).
- **Audio callback thread**: speaker output callback (`NonBlockingAudioPlayer._audio_callback()`).
- **Cloned TTS synthesis thread (optional)**: streaming/cancellation worker in `abstractvoice/vm/tts_mixin.py`.

## Offline-first model policy

The library defaults to `allow_downloads=True`, but the REPL creates `VoiceManager(..., allow_downloads=False)` (see `abstractvoice/examples/cli_repl.py`).

Explicit prefetch entry points:
- `python -m abstractvoice download ...` (`abstractvoice/__main__.py`)
- `abstractvoice-prefetch ...` (`abstractvoice/prefetch.py`)

See `docs/installation.md` and `docs/model-management.md`.

## Optional: AbstractCore plugin integration

When installed alongside `abstractcore`, AbstractVoice exposes a capability plugin via the entry point:

- `pyproject.toml` → `[project.entry-points."abstractcore.capabilities_plugins"]`
- Implementation: `abstractvoice/integrations/abstractcore_plugin.py`

It provides:
- a voice backend (TTS+STT) that can optionally store generated audio into an `artifact_store`
- an audio backend (STT) for transcription-only use

This is not required for using AbstractVoice as a standalone library.
