# Architecture

AbstractVoice `0.10.x` is built around a small public facade and engine adapters
that keep optional heavy runtimes out of the default path.

Use `docs/api.md` for the supported integrator contract, `docs/repl_guide.md`
for REPL behavior and commands, and `docs/known-issues.md` for current release
caveats.

For acronyms used here (TTS/STT/VAD/VM/MM), see `docs/acronyms.md`.

## TL;DR

- `abstractvoice.VoiceManager` is the orchestration façade (`abstractvoice/vm/*`).
- **TTS (default)**: TTS adapter registry resolves `openai` / `auto` to `OpenAICompatibleTTSAdapter`; local opt-in engines include Piper and Supertonic 3; playback uses `AdapterTTSEngine` -> `NonBlockingAudioPlayer` when local audio output is requested.
- **STT (default)**: `openai` / `auto` routes `transcribe_*()` and `listen()` recognition to `OpenAICompatibleSTTAdapter`. Local microphone capture still uses `VoiceRecognizer` → `VoiceDetector` and can pass captured audio to the selected STT adapter.
- **Voice cloning (optional)**: `VoiceCloner` + clone store + engine backends (`omnivoice|f5_tts|chroma|audiodit|openai|openai-compatible`); new local clones default to OmniVoice.
- Voice modes are implemented by wiring TTS playback callbacks to recognizer controls (`abstractvoice/vm/core.py`).

## Component diagram

```mermaid
flowchart LR
  App[Your app / REPL] <--> VM[VoiceManager]

  VM -->|speak()*| TTSEngine[AdapterTTSEngine]
  TTSEngine -->|synthesize| TTSAdapter[TTSAdapter]
  TTSAdapter --> Piper[PiperTTSAdapter]
  TTSAdapter --> Supertonic[SupertonicTTSAdapter]
  TTSAdapter --> RemoteTTS[OpenAICompatibleTTSAdapter]
  TTSAdapter --> AudioDiT[AudioDiTTTSAdapter]
  TTSAdapter --> OmniVoice[OmniVoiceTTSAdapter]
  TTSEngine --> Player[NonBlockingAudioPlayer]
  Player --> Out[(sounddevice OutputStream)]

  VM -->|listen()*| Rec[VoiceRecognizer]
  Rec --> In[(sounddevice InputStream)]
  Rec --> VAD[VoiceDetector (webrtcvad)]
  Rec --> STT[FasterWhisperAdapter]
  Rec --> RemoteSTT[OpenAICompatibleSTTAdapter]

  Player -. on_audio_chunk (optional AEC) .-> Rec
```

## Code map (evidence)

Start points (in call order):

- Public façade: `abstractvoice/voice_manager.py` → `abstractvoice/vm/manager.py`
- TTS orchestration: `abstractvoice/vm/tts_mixin.py`
- STT/listening orchestration: `abstractvoice/vm/stt_mixin.py`
- Playback/lifecycle wiring + voice modes: `abstractvoice/vm/core.py`

TTS implementation:

- TTS adapter interface: `abstractvoice/adapters/base.py`
- Piper adapter: `abstractvoice/adapters/tts_piper.py`
- Supertonic adapter/runtime: `abstractvoice/adapters/tts_supertonic.py`, `abstractvoice/supertonic/runtime.py`
- Remote OpenAI-compatible TTS adapter: `abstractvoice/adapters/tts_openai_compatible.py` + `abstractvoice/adapters/openai_compatible_http.py`
- TTS engine selection (registry): `abstractvoice/adapters/tts_registry.py`
- Local model presence (filesystem only, no engine imports): `abstractvoice/local_models.py`
- AudioDiT adapter/runtime: `abstractvoice/adapters/tts_audiodit.py`, `abstractvoice/audiodit/runtime.py`
- OmniVoice adapter/runtime: `abstractvoice/adapters/tts_omnivoice.py`, `abstractvoice/omnivoice/runtime.py`
- TTS engine wrapper (back-compat contract): `abstractvoice/tts/adapter_tts_engine.py`
- Low-latency audio player: `abstractvoice/tts/tts_engine.py`

STT implementation:

- Mic/VAD/STT loop: `abstractvoice/recognition.py`
- VAD wrapper: `abstractvoice/vad/voice_detector.py`
- faster-whisper adapter: `abstractvoice/adapters/stt_faster_whisper.py`
- Remote OpenAI-compatible STT adapter: `abstractvoice/adapters/stt_openai_compatible.py`
- Stop phrase normalization: `abstractvoice/stop_phrase.py`

Optional features:

- AEC (extra): `abstractvoice/aec/webrtc_apm.py` (used by `abstractvoice/recognition.py`)
- Voice cloning (extra): `abstractvoice/cloning/*` (manager/engines/store; used by `abstractvoice/vm/tts_mixin.py`)
- Remote cloning bridge: `abstractvoice/cloning/engine_remote.py`
- AbstractCore plugin: `abstractvoice/integrations/abstractcore_plugin.py`

## Data flows

### TTS (playback)

1) Your app calls `VoiceManager.speak()` (`abstractvoice/vm/tts_mixin.py`).
2) Default path (OpenAI remote): `AdapterTTSEngine.speak()` (`abstractvoice/tts/adapter_tts_engine.py`)
   - synthesizes audio via the selected adapter (default `OpenAICompatibleTTSAdapter`, `abstractvoice/adapters/tts_openai_compatible.py`)
   - enqueues audio into `NonBlockingAudioPlayer.play_audio()` (`abstractvoice/tts/tts_engine.py`)
3) Playback runs in the PortAudio callback thread (`sounddevice.OutputStream`).

Pause/resume is implemented by toggling a lock-protected paused flag inside the audio callback (see `NonBlockingAudioPlayer.pause()/resume()`).

### STT (microphone listening)

1) Your app calls `VoiceManager.listen()` (`abstractvoice/vm/stt_mixin.py`).
2) A `VoiceRecognizer` instance is created (`abstractvoice/recognition.py`) with:
   - a VAD (`VoiceDetector`, `abstractvoice/vad/voice_detector.py`)
   - the configured STT adapter (OpenAI remote by default; `FasterWhisperAdapter` when explicitly selected)
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

## Remote-first and offline model policy

The library default uses OpenAI remote audio and requires `OPENAI_API_KEY` or
`remote_api_key=...`; it does not download local model weights. Local engines
are explicit (`tts_engine="piper"`, `tts_engine="supertonic"`,
`stt_engine="faster_whisper"`, etc.) and
respect `allow_downloads`. The REPL creates `VoiceManager(...,
allow_downloads=False)` so local engines never fetch weights implicitly.
The CLI and web examples use an additional install-aware TTS resolver:
installed Supertonic first, installed Piper second, then OpenAI remote. That
keeps plain `abstractvoice` remote/OpenAI while full local install profiles
such as `abstractvoice[all-apple]` and `abstractvoice[all-gpu]` launch on
Supertonic without changing the library/server `VoiceManager()` default.

Runtime base-TTS switching goes through `VoiceManager.set_tts_engine(...)`.
That replaces the adapter, rewires the playback wrapper, and resets the active
profile to the default for the engine/language so stale Piper/Supertonic/remote
voice selections do not bleed across engine changes.
`AdapterTTSEngine.warmup_audio_output()` can open the playback stream before
the first utterance; `NonBlockingAudioPlayer` prefers the hardware default
output sample rate and resamples synthesized audio when needed, avoiding slow
first-call CoreAudio negotiation on macOS.

Discovery follows the same policy from the other direction: asking which providers and models are
available is a filesystem question, answered by `abstractvoice/local_models.py` without importing an
engine runtime or loading weights. A local engine is discoverable when its runtime is importable and
at least one of its models is cached; engines that download Hugging Face repos are matched against
the shared HF cache, while `piper` and `supertonic` expose their own cache probes
(`cached_piper_model_ids`, `cached_piper_voice_profiles`, `is_supertonic_cached`). Remote providers
are probed concurrently under a single discovery budget, so a listing costs the slowest provider
rather than the sum of their timeouts, and a provider that does not answer is reported as unreachable
rather than as one with nothing to offer.

Explicit prefetch entry points:
- `python -m abstractvoice download ...` (`abstractvoice/__main__.py`)
- `abstractvoice-prefetch ...` (`abstractvoice/prefetch.py`)

Supertonic 3 is implemented as an internal ONNX adapter, not through
Supertone's Python SDK. Its artifacts are fetched from
`Supertone/supertonic-3` into `~/.cache/abstractvoice/supertonic-3` with
`--supertonic`; profile metadata (`M1`-`M5`, `F1`-`F5`) is available without
loading or downloading the model.

See `docs/installation.md` and `docs/model-management.md`.

## Optional: AbstractCore plugin integration

When installed alongside `abstractcore`, AbstractVoice exposes a capability plugin via the entry point:

- `pyproject.toml` → `[project.entry-points."abstractcore.capabilities_plugins"]`
- Implementation: `abstractvoice/integrations/abstractcore_plugin.py`
  - Artifact store adapter (AbstractRuntime-compatible, duck-typed): `abstractvoice/artifacts.py`

It provides:
- a voice backend (TTS+STT) that can optionally store generated audio into an `artifact_store`
- an audio backend (STT) for transcription-only use

The local FastAPI web UI (`abstractvoice web`) is a small example wrapper around
`VoiceManager` for quick browser testing. Its local `/api/*` routes map directly
to `VoiceManager` functions and include browser-only conveniences such as
assistant/user voice defaults. It also includes an example-only LLM bridge that
forwards one OpenAI-compatible chat request to a local provider such as Ollama;
the `/v1/audio/*` routes are smoke-test aliases.

AbstractCore owns the production HTTP server surface. When AbstractCore Server
is installed and running, these capability backends can power OpenAI-compatible
endpoints such as `POST /v1/audio/speech` and `POST /v1/audio/transcriptions`.

This is not required for using AbstractVoice as a standalone library.
