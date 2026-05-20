# API

This is the supported integrator contract for AbstractVoice.

Start with `README.md` and `docs/getting-started.md` for setup. Use
`docs/faq.md` for cache/history reset and troubleshooting, `docs/repl_guide.md`
for the interactive REPL, and `docs/architecture.md` for implementation details.

Implementation map:
- `abstractvoice/voice_manager.py` → `abstractvoice/vm/manager.py` (constructor + wiring)
- `abstractvoice/vm/tts_mixin.py` (TTS + cloning methods)
- `abstractvoice/vm/stt_mixin.py` (STT + listening methods)
- `abstractvoice/vm/core.py` (voice-mode behavior during playback)

## Primary entry point

- `abstractvoice.VoiceManager`
- `abstractvoice.VoiceProfile` (data type; used by the voice-profile APIs)

```python
from abstractvoice import VoiceManager

vm = VoiceManager(language="en", remote_api_key="sk-...", allow_downloads=True)
```

## Constructor (most-used knobs)

The source of truth is `abstractvoice/vm/manager.py`:

```python
VoiceManager(
    language: str = "en",
    tts_model: str | None = None,
    whisper_model: str = "base",
    debug_mode: bool = False,
    tts_engine: str = "openai",
    stt_engine: str = "openai",
    allow_downloads: bool = True,
    cloned_tts_streaming: bool = True,
    cloning_engine: str = "omnivoice",
    tts_delivery_mode: str | None = None,  # buffered|streamed (override)
    stt_model: str | None = None,
    remote_base_url: str | None = None,
    remote_api_key: str | None = None,
    remote_timeout_s: float | None = None,
)
```

Notes:
- `VoiceManager()` and library `auto` are remote-first. Hosted OpenAI audio requires `OPENAI_API_KEY` or `remote_api_key=...`.
- The CLI and web examples deliberately wrap this with an interactive TTS
  resolver: installed Supertonic, installed Piper, then OpenAI remote. Plain
  `abstractvoice` starts on OpenAI; `abstractvoice[all-apple]` and
  `abstractvoice[all-gpu]` start on Supertonic. That behavior lives in
  `abstractvoice/examples/tts_defaults.py` and does not change the library
  constructor.
- `allow_downloads` gates *implicit* local model downloads in adapters. The REPL sets `False` (offline-first).
- `whisper_model` controls the faster-whisper model size used by local `listen()` / `transcribe_*()` paths.
- `tts_engine` selects the TTS provider and supports:
  - `openai` (default; remote OpenAI `/v1/audio/speech`; requires `OPENAI_API_KEY`)
  - `auto` (deterministic default: resolves to `openai`)
  - `piper` (local TTS; requires `abstractvoice[piper]`, `abstractvoice[apple]`, or `abstractvoice[gpu]`)
  - `supertonic` (local fixed-profile ONNX TTS; requires `abstractvoice[supertonic]`, `abstractvoice[apple]`, or `abstractvoice[gpu]`)
  - `openai-compatible` (remote compatible `/v1/audio/speech`; configure `remote_base_url` or `OPENAI_BASE_URL`)
  - `audiodit` (LongCat-AudioDiT; requires `abstractvoice[audiodit]`; upstream focuses on EN/ZH; direct/base TTS has a known quality caveat in `0.8.1`)
  - `omnivoice` (OmniVoice; requires `abstractvoice[omnivoice]`; upstream supports 600+ languages)
- `stt_engine` selects the STT provider and supports `openai|auto|faster_whisper|openai-compatible|transformers-asr`. `auto` resolves to `openai`.
  - `faster_whisper` requires `abstractvoice[stt]`, `abstractvoice[apple]`, or `abstractvoice[gpu]`, and uses `whisper_model`/`--whisper` for `tiny|base|small|medium|large-v2|large-v3|large`.
  - `transformers-asr` requires `abstractvoice[stt-hf]`, `abstractvoice[apple]`, or `abstractvoice[gpu]`, and uses `stt_model` as the Hugging Face model id (for example `openai/whisper-large-v3`, `openai/whisper-large-v3-turbo`, or `Qwen/Qwen3-ASR-1.7B`).
  Missing credentials or missing explicit local dependencies raise actionable errors; the legacy OpenAI Whisper fallback was removed.
- `tts_model` is reserved/back-compat for local Piper (selection is language-driven today); for remote TTS it maps to the request `model`.
- For remote STT, `stt_model` maps to the transcription request `model`.
- For `stt_engine="transformers-asr"`, `stt_model` is the Hugging Face model id to load.
- Remote configuration can be passed in the constructor or via env vars:
  - OpenAI: `OPENAI_API_KEY`, optional `ABSTRACTVOICE_OPENAI_TTS_MODEL`, `ABSTRACTVOICE_OPENAI_STT_MODEL`
  - Compatible endpoints: `OPENAI_BASE_URL`, optional `OPENAI_API_KEY`, `ABSTRACTVOICE_REMOTE_TTS_MODEL`, `ABSTRACTVOICE_REMOTE_STT_MODEL`
  - OpenAI-compatible profile discovery: `GET /audio/voices` is tried by default for compatible endpoints; override with `ABSTRACTVOICE_REMOTE_VOICE_PROFILE_PATH` or `ABSTRACTVOICE_REMOTE_VOICE_PROFILE_PATHS`. Static voice/profile ids can also be supplied with `ABSTRACTVOICE_REMOTE_TTS_VOICES`.
- `tts_delivery_mode` is an optional override that applies consistently to both base TTS and cloned voices:
  - `buffered`: synthesize full audio first (one payload)
  - `streamed`: deliver audio in chunks when available (lower time-to-first-audio)
- `cloning_engine` defaults to `omnivoice` for new local clones. Install
  `abstractvoice[omnivoice]` or a platform/full profile for that default, or
  pass `f5_tts`, `chroma`, `audiodit`, `openai`, or `openai-compatible`
  explicitly.

Supported language codes for the Piper mapping: `en, fr, de, es, ru, zh` (see `abstractvoice/config/voice_catalog.py` and `abstractvoice/adapters/tts_piper.py`).
Supertonic supports fixed-style local TTS for `ar, bg, cs, da, de, el, en, es, et, fi, fr, hi, hr, hu, id, it, ja, ko, lt, lv, nl, pl, pt, ro, ru, sk, sl, sv, tr, uk, vi`.
For other non-Piper providers (e.g. OmniVoice or remote OpenAI-compatible providers), `language` is treated as a pass-through hint and the provider decides what it supports.

## TTS (text → audio)

- `speak(text: str, speed: float = 1.0, callback=None, voice: str | None = None, *, sanitize_syntax: bool = True) -> bool`
  - Plays audio locally (non-blocking playback; synthesis time depends on backend).
  - If `voice` is provided, it is treated as a cloned `voice_id` (requires a cloning backend extra such as `abstractvoice[omnivoice]`; `abstractvoice[cloning]` is the explicit OpenF5 backend).
  - By default, common Markdown syntax is stripped from spoken output (headers + emphasis). Set `sanitize_syntax=False` to speak raw text.

- `set_speed(speed: float) -> bool`, `get_speed() -> float`
  - Adjusts the default speaking speed used by `speak_to_*()` and the REPL.

- `set_tts_quality_preset(preset: str) -> bool`, `get_tts_quality_preset() -> str | None`
  - Provider-agnostic speed/quality knob (`low|standard|high`). Back-compat aliases: `fast`→`low`, `balanced`→`standard`.
  - Providers that don’t support quality tuning may return `False` / `None` (Piper is typically a no-op).
  - For AudioDiT this primarily maps to diffusion `steps` (and a small guidance-strength tweak).
  - For Supertonic this maps to ONNX vector-estimator steps: low `5`, standard `8`, high `12`.

- `get_profiles(*, kind: str = "tts") -> list[VoiceProfile]`
- `set_profile(profile_id: str, *, kind: str = "tts") -> bool`
- `get_active_profile(*, kind: str = "tts") -> VoiceProfile | None`
- `set_tts_engine(engine: str, *, tts_model: str | None = None) -> str`
  - Cross-provider **voice profile** abstraction (preset packs).
  - Profiles are **provider-local**: select `tts_engine` (provider) first, then apply a profile id for that provider.
  - Runtime provider switching should use `set_tts_engine(...)`; it rebuilds the base TTS adapter, rewires playback callbacks, and resets the active profile to the provider/language default (for example Piper `en` -> `amy`, Supertonic -> `M1`, OpenAI -> `alloy`).
  - Providers without profiles return an empty list / False / None.
  - **Concurrency note**: profile selection mutates provider state. For servers, prefer one `VoiceManager` per session (or guard profile changes with a lock).
  - **Remote OpenAI note**: hosted built-in voices are always exposed as profiles (for example `vm.set_profile("alloy")`), and the adapter also tries OpenAI voice discovery for account/org-specific voices such as `voice_...`. `tts_engine="openai"` defaults to `https://api.openai.com/v1` and reads `OPENAI_API_KEY`.
  - **Remote compatible note**: compatible endpoints may expose `GET /v1/audio/voices` (adapter path: `GET /audio/voices`) returning `profiles`, `voices`, `cloned_voices`, or OpenAI-style `data`. Returned ids are exposed as `VoiceProfile`s and used as the request `voice` for `/audio/speech`.
  - The `voice=` argument on `speak_to_bytes(...)` remains the cloned-voice handle path for backward compatibility; select base-provider voices with `set_profile(...)`.
  - **Supertonic note**: `tts_engine="supertonic"` exposes fixed local profiles `M1`-`M5` and `F1`-`F5`. Listing or selecting profiles does not download the model; synthesis requires cached artifacts or `allow_downloads=True`.
  - **OmniVoice notes**:
    - Some profiles may enable **persistent prompt caching** (a tokenized `voice_clone_prompt`). The first `set_profile(...)` can pay a one-time build cost; later synthesis reuses cached tokens for stable voice identity. Prompt-conditioned synthesis can be heavier than pure voice design; use `/tts quality low|standard|high` (or `VoiceManager.set_tts_quality_preset(...)`) to tune the trade-off.
    - On macOS / Apple Silicon, OmniVoice uses **MPS (Metal)** by default when `device="auto"`.

- `pause_speaking() -> bool`, `resume_speaking() -> bool`, `stop_speaking() -> bool`
  - Playback control.

- `is_speaking() -> bool`, `is_paused() -> bool`
  - Playback state helpers.

- `set_tts_delivery_mode(mode: str | None) -> bool`, `get_tts_delivery_mode() -> str`, `get_tts_delivery_modes() -> dict`
  - Toggle buffered vs streamed delivery (applies to both base TTS and cloned voices).
  - **Behavior note**: streamed delivery is implemented as a pipeline:
    - **text** is chunked into short segments (sentence-first),
    - then each segment is synthesized and enqueued as soon as possible.
    - Engines that can stream audio natively may further reduce TTFB by yielding multiple audio chunks per segment.

- `speak_to_bytes(text: str, format: str = "wav", voice: str | None = None, *, sanitize_syntax: bool = True) -> bytes`
  - Headless/server‑friendly: returns encoded audio bytes.

- `speak_to_audio_chunks(text: str, *, voice: str | None = None, sanitize_syntax: bool = True) -> Iterator[tuple[np.ndarray, int]]`
  - Headless/server‑friendly: yields `(audio_chunk, sample_rate)` tuples for incremental delivery.

- `open_tts_text_stream(*, voice: str | None = None, callback=None, sanitize_syntax: bool = True, max_chars: int | None = None, min_chars: int | None = None) -> TextToSpeechStream`
  - Push-based streaming bridge for **LLM streaming → TTS streaming** pipelining.
  - Returned object supports: `.push(delta)`, `.close()`, `.cancel()`, `.join(timeout=...)`.

- `speak_to_file(text: str, output_path: str, format: str | None = None, voice: str | None = None, *, sanitize_syntax: bool = True) -> str`
  - Writes an audio file and returns the path.

### Language & voice selection

- `set_language(language: str) -> bool`
  - Switches the active language.
  - For explicit Piper, validation uses the curated Piper mapping in `abstractvoice/config/voice_catalog.py`.
  - For Supertonic, validation uses the adapter's 31-language ONNX text frontend list.
  - For non-Piper providers such as OmniVoice, the language code is passed through to the adapter and the provider decides what it supports.
  - If microphone listening is active, the recognizer is recreated on the next `listen(...)` call so STT receives the updated language.

- `get_language() -> str`, `get_language_name(language_code: str | None = None) -> str`

- `get_supported_languages() -> list[str]`

- `list_available_models(language: str | None = None) -> dict`
  - Lists voice/model catalog entries for CLI/web display (see `abstractvoice/vm/tts_mixin.py`).
  - Piper returns local voice cache status by language.
  - Supertonic returns the 10 fixed styles for the active language plus shared cache status.
  - OpenAI/OpenAI-compatible TTS returns remote voice profiles plus configured/discovered TTS model ids when the active adapter supports model listing.
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

Requires installing at least one cloning backend extra (and explicit artifact
downloads; see `docs/installation.md`). The recommended/default local backend
is OmniVoice:

- `abstractvoice[omnivoice]` → `omnivoice` (default for new clones)
- `abstractvoice[cloning]` → `f5_tts`
- `abstractvoice[chroma]` → `chroma`
- `abstractvoice[audiodit]` → `audiodit`

Supertonic is not listed here because it is fixed-profile base TTS, not a
voice-cloning provider.

Remote clone-compatible endpoints can also be used without local cloning model
weights by selecting `cloning_engine="openai-compatible"` (or
`engine="openai-compatible"` per call). Configure `remote_base_url` or
`OPENAI_BASE_URL`; the default clone endpoint is `POST /voice/clone`
and must return a remote voice id (`voice_id` or `id`). The local clone store
keeps a handle and routes later `speak_to_bytes(..., voice=<local_id>)` calls to
remote `/audio/speech` with that remote voice id.

`cloning_engine="openai"` targets OpenAI's hosted API by default. Custom voice
creation is provider/org gated and requires explicit consent configuration such
as `ABSTRACTVOICE_OPENAI_VOICE_CONSENT_ID`; otherwise the adapter raises an
actionable error instead of silently pretending cloning is standardized.

Core cloning calls:

- `clone_voice(reference_audio_path: str, name: str | None = None, *, reference_text: str | None = None, engine: str | None = None) -> str`
- `clone_voice_from_wav_bytes(wav_bytes: bytes, name: str | None = None, *, reference_text: str | None = None, engine: str | None = None) -> str`
- `speak(..., voice="<voice_id>")` / `speak_to_bytes(..., voice="<voice_id>")` / `speak_to_file(..., voice="<voice_id>")`
- `list_cloned_voices()`, `get_cloned_voice(voice_id: str) -> dict`

Clone management helpers:

- `set_cloned_voice_reference_text(voice_id: str, reference_text: str) -> bool`
- `rename_cloned_voice(voice_id: str, new_name: str) -> bool`
- `delete_cloned_voice(voice_id: str) -> bool`
- `export_voice(voice_id: str, path: str) -> str`, `import_voice(path: str) -> str`
- `set_cloned_tts_quality(preset: str) -> bool` (`low|standard|high`; aliases: `fast`, `balanced`)
- `get_cloning_runtime_info() -> dict`
- `unload_cloning_engines(*, keep_engine: str | None = None) -> int` (best-effort memory relief)
- `unload_piper_voice() -> bool` (best-effort memory relief)

For the user-facing workflow and commands, see `docs/repl_guide.md`.

Provider caveats that affect release choice are tracked in `docs/known-issues.md`.

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
python -m abstractvoice download --stt-hf openai/whisper-large-v3
python -m abstractvoice download --piper en
python -m abstractvoice download --supertonic # optional; requires abstractvoice[supertonic]
python -m abstractvoice download --openf5   # optional; requires abstractvoice[cloning]
python -m abstractvoice download --chroma   # optional; requires abstractvoice[chroma] (GPU-heavy)
python -m abstractvoice download --audiodit # optional; requires abstractvoice[audiodit]
python -m abstractvoice download --omnivoice # optional; requires abstractvoice[omnivoice]
```

Or use the convenience entrypoint:

```bash
abstractvoice-prefetch --stt small
abstractvoice-prefetch --stt-hf openai/whisper-large-v3
abstractvoice-prefetch --piper en
abstractvoice-prefetch --supertonic         # optional; requires abstractvoice[supertonic]
abstractvoice-prefetch --openf5            # optional; requires abstractvoice[cloning]
abstractvoice-prefetch --chroma            # optional; requires abstractvoice[chroma] (GPU-heavy)
abstractvoice-prefetch --audiodit          # optional; requires abstractvoice[audiodit]
abstractvoice-prefetch --omnivoice         # optional; requires abstractvoice[omnivoice]
```

Notes:
- `--chroma` artifacts may require Hugging Face access to download.

See also: `docs/installation.md`, `docs/model-management.md`, and `docs/voices-and-licenses.md`.

## Performance note: prefetch vs preload (important for servers)

- **Prefetch** (download to disk): `python -m abstractvoice download ...` / `abstractvoice-prefetch ...`
- **Preload** (load into memory): create a **long-lived** `VoiceManager` (or adapter) and reuse it.

If you construct a new `VoiceManager` for every request, heavy engines (AudioDiT/OmniVoice) will pay a large one-time cost repeatedly (imports + weight load + accelerator kernel compilation).

Recommended pattern (server/process startup):

```python
from abstractvoice import VoiceManager

# Load once, reuse for all requests.
vm = VoiceManager(
    language="en",
    tts_engine="omnivoice",
    stt_engine="openai",
    remote_api_key="sk-...",
    allow_downloads=False,
)
```

## Integrations (AbstractFramework ecosystem)

AbstractVoice is designed to work standalone, and also integrate cleanly into the AbstractFramework ecosystem (AbstractCore + AbstractRuntime). Overview and links: `README.md`.

Boundary note:
- AbstractVoice owns the in-process voice backend (`VoiceManager`, adapters, model/cache policy).
- AbstractCore owns agent orchestration, provider routing, capability selection, and OpenAI-compatible HTTP endpoints.
- When both are installed, AbstractCore can expose AbstractVoice-backed audio endpoints such as `POST /v1/audio/speech` and `POST /v1/audio/transcriptions`.

### AbstractCore capability plugin (auto-discovery)

AbstractVoice exposes an AbstractCore capability plugin entry point:

- Entry point declaration: `pyproject.toml` → `[project.entry-points."abstractcore.capabilities_plugins"]`
- Implementation: `abstractvoice/integrations/abstractcore_plugin.py`

The plugin registers:
- a voice backend (`backend_id="abstractvoice:default"`) for TTS+STT
- an audio backend (`backend_id="abstractvoice:stt"`) for STT-only

Audio outputs can optionally be stored into an AbstractRuntime-like `artifact_store` via the duck-typed adapter in `abstractvoice/artifacts.py`.

The voice backend also exposes thin catalog discovery methods for Core/Gateway
integration code:
- `list_profiles(kind="tts") -> list[dict]`
- `list_models(kind="tts"|"stt"|"cloning", provider: str | None = None) -> list[str]`
- `list_tts_models(provider: str | None = None) -> list[str]`
- `list_stt_models(provider: str | None = None) -> list[str]`
- `list_cloning_models(provider: str | None = None) -> list[str]`
- `list_tts_voices(provider: str | None = None, model: str | None = None, include_clones: bool = True) -> list[dict]`
- `list_cloned_voices(provider: str | None = None, model: str | None = None) -> list[dict]`
- `list_voices(...) -> list[dict]` (alias of `list_tts_voices(...)`)
- `available_providers() -> {tts, stt, cloning, providers, details}`
- `compatibility_catalog() -> {version, providers}`
- `get_capability_support(kind, feature, provider, model=None, surface="default") -> dict | None`
- `find_compatible_models(kind, feature, surface="default", support_in=("native","emulated","conditional")) -> list[dict]`
- `voice_catalog() -> {kind, provider_id (alias engine_id), active_profile, active_model, voices (profiles + clones), tts_models, stt_models, tts_models_by_provider, stt_models_by_provider, tts_model_variants, stt_engine_variants, tts_catalog_by_provider, stt_catalog_by_provider, available_providers, catalog}`
- `voice_catalog()` also includes additive capability metadata:
  - `controls`: legacy control map kept backward-compatible for existing clients
  - `tts_capabilities`: richer per-field support truth (`native|emulated|conditional|unsupported`) for newer clients
  - `speech_request_contract`: current request contract id (currently `"speech_request_v1"`)
  - `compatibility_catalog`: package-owned provider/model feature matrix across `tts`, `stt`, and `cloning`
- `tts(text, *, provider=None, model=None, voice=None, format="wav", ...) -> bytes | artifact_ref`
- `stt(audio, *, provider=None, model=None, language=None, ...) -> str`
- `transcribe(audio, *, provider=None, model=None, language=None, ...) -> str`

These methods delegate to the active `VoiceManager` and keep voice/profile/model
semantics in AbstractVoice. AbstractCore still owns HTTP routing, auth, and
browser/security policy.

`available_providers()` is intentionally lightweight: it lists currently usable
provider ids without constructing local model runtimes. Remote OpenAI appears
when `OPENAI_API_KEY` is configured or an active/injected manager is already
using it; OpenAI-compatible appears when a compatible base URL is configured.
Local providers appear when their runtime dependency is installed and any required
cached model artifacts are present. The same payload also includes
`known_tts_providers`, `known_stt_providers`, and `known_cloning_providers` so
UI/provider selectors can distinguish installed availability from the broader
capability catalog.

The provider/model/voice abstraction used by the plugin is:
- `provider`: backend/engine id such as `openai`, `openai-compatible`,
  `piper`, `supertonic`, `faster-whisper`, or `transformers-asr`
- `model`: provider-specific selectable model id
- `voice`: base voice/profile id or cloned voice id available for the selected
  `provider` + `model`

`voice_catalog()` is meant to be the single rich discovery payload for
AbstractCore/Gateway:
- `tts_catalog_by_provider[provider]` contains `models`, `model_variants`,
  `voices`, `profiles`, `cloned_voices`, `voices_by_model`, and `formats`
- `stt_catalog_by_provider[provider]` contains `models`, `model_variants`, and
  `formats`
- `controls` remains the legacy compatibility surface for control discovery
  and still uses simple `supported` booleans for fields such as `speed`,
  `quality_preset`, `instructions`, `profile`, and `voice_clone`
- `tts_capabilities` is the newer truthful capability map for engine-aware
  clients and reports `support` plus optional `reason` per field
  - support values may be `native`, `emulated`, `conditional`, or `unsupported`
  - `conditional` means AbstractVoice forwards or orchestrates the feature, but
    actual backend support can still vary by provider or endpoint
- `speech_request_contract` versions the richer package-owned request shape
  without forcing older clients to understand it
- `compatibility_catalog` is the central matrix for provider/model capability
  queries across `tts`, `stt`, and `cloning`
  - each provider entry can define provider-wide defaults plus explicit model
    records
  - feature support is surface-specific, for example `tts.bytes` vs
    `tts.playback`

The lighter helper methods are convenience views over that same abstraction:
- `available_providers()` for provider selectors
- `list_models(...)` / `list_tts_models(...)` / `list_stt_models(...)` /
  `list_cloning_models(...)` for model selectors
- `list_tts_voices(...)` / `list_cloned_voices(...)` for voice selectors
- `get_capability_support(...)` / `find_compatible_models(...)` for
  provider-model-surface capability filtering across `tts`, `stt`, and
  `cloning`

For both TTS and STT, the plugin accepts either split fields or combined
`provider:model` selectors. AbstractCore callers may pass separate arguments
such as `provider="openai", model="tts-1"` or
`provider="transformers-asr", model="Qwen/Qwen3-ASR-1.7B"`, or a single
combined provider value such as `provider="openai:tts-1"`,
`provider="faster-whisper:large"`, or
`provider="transformers-asr:Qwen/Qwen3-ASR-1.7B"`.

For TTS execution, the plugin accepts either a simple `voice="alloy"` string or
the selected voice/cloned-voice record from the catalog. When a voice dict is
passed, the plugin derives `provider` and `model` from that record when needed.

Plugin configuration (owner `config` dict, best-effort). In AbstractCore
integrations, the env/default path uses OpenAI remote TTS/STT
(`OPENAI_API_KEY`, optional `ABSTRACTVOICE_OPENAI_*` overrides) unless owner
config or `ABSTRACTVOICE_TTS_ENGINE` / `ABSTRACTVOICE_STT_ENGINE` selects a
different provider:
- `voice_language`: default language (e.g. `"en"`)
- `voice_allow_downloads`: allow on-demand downloads (bool)
- `voice_tts_engine`: base TTS provider (`"auto"|"piper"|"supertonic"|"openai"|"openai-compatible"|"audiodit"|"omnivoice"`)
- `voice_stt_engine`: STT provider (`"auto"|"faster_whisper"|"transformers-asr"|"openai"|"openai-compatible"`)
- `voice_tts_model`: model id for remote TTS providers
- `voice_stt_model`: model id for remote STT providers, or a Hugging Face model id when `voice_stt_engine="transformers-asr"`
- `voice_remote_base_url`: base URL for OpenAI-compatible remote audio endpoints
- `voice_remote_api_key`: optional bearer key for remote audio endpoints
- `voice_remote_timeout_s`: request timeout for remote audio endpoints
- `voice_whisper_model`: faster-whisper model size (e.g. `"base"`, `"small"`)
- `voice_cloning_engine`: default cloning provider (`"omnivoice"` by default; also `"f5_tts"|"chroma"|"audiodit"|"openai"|"openai-compatible"`)
- `voice_cloned_tts_streaming`: stream cloned-voice chunks for faster time-to-first-audio (bool). Used when `voice_tts_delivery_mode` is unset.
- `voice_tts_delivery_mode`: unified audio delivery mode for base + cloned voices (`"buffered"|"streamed"`). Takes precedence over `voice_cloned_tts_streaming`.
- `voice_tts_streaming`: bool alias for `voice_tts_delivery_mode` (`true` → `"streamed"`, `false` → `"buffered"`).
- `voice_debug_mode`: enable debug prints (bool)

Boolean owner config/env values accept common strings such as `true`, `false`,
`on`, `off`, `1`, and `0`; string values like `"false"` are not treated as
truthy.

Performance note:
- The capability plugin caches `VoiceManager` instances **in-process** (keyed by the config above) so provider runtimes are **not reloaded per request**.

TTS metrics:
- After synthesis, the plugin stores best-effort stats in artifact metadata under `abstractvoice_tts` (when `artifact_store` is used).

### AbstractCore OpenAI-compatible audio endpoints

The production OpenAI-compatible HTTP server lives in AbstractCore. AbstractVoice
also ships a local FastAPI web example (`abstractvoice web`) for package-level
smoke testing, but the supported API server path is AbstractCore Server.

With `abstractcore[server]` and `abstractvoice` installed in the same
environment, AbstractCore delegates its audio endpoints to the discovered
capability plugin:

- `POST /v1/audio/speech` -> `core.voice.tts(...)` -> `VoiceManager.speak_to_bytes(...)`
- `POST /v1/audio/transcriptions` -> `core.audio.transcribe(...)` -> `VoiceManager.transcribe_*()`
- `GET /v1/audio/voices` -> `core.voice.voice_catalog()` / `list_tts_voices(...)` for provider, model, profile, and cloned-voice discovery
- `GET /v1/audio/speech/models` -> `core.voice.list_tts_models(...)` or `voice_catalog()["tts_catalog_by_provider"]`
- `GET /v1/audio/transcriptions/models` -> `core.voice.list_stt_models(...)` or `voice_catalog()["stt_catalog_by_provider"]`

Recommended AbstractCore mapping:

```python
providers = core.voice.available_providers()
catalog = core.voice.voice_catalog()

tts_audio = core.voice.tts(
    "Hello.",
    provider="openai",
    model="tts-1",
    voice="alloy",
    format="wav",
)

text = core.audio.transcribe(
    wav_bytes,
    provider="faster-whisper",
    model="large",
    language="en",
)
```

That same shape also supports local or remote cloned voices: the selected
catalog item from `tts_catalog_by_provider[provider]["cloned_voices"]` can be
passed back as the TTS `voice`, or its `voice_id` can be passed directly.

Example:

```bash
OPENAI_API_KEY=... python -m abstractcore.server.app

curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"Hello.","format":"wav"}' \
  --output hello.wav

curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@hello.wav" \
  -F "language=en"
```

If AbstractCore Server is configured with `ABSTRACTCORE_API_KEY`, include the
standard `Authorization: Bearer <key>` header. `ABSTRACTCORE_API_KEY` is
AbstractCore’s master key: when set, AbstractCore can use any preconfigured
provider API keys, while provider-specific env vars (for example
`OPENAI_API_KEY`) can still override.

If the plugin is unavailable,
AbstractCore returns `501` with install/config guidance instead of silently
falling back.

For `openai-compatible` plugin configuration, do not set `voice_remote_base_url`
to the same AbstractCore Server instance that is currently routing the
`/v1/audio/*` request. That configuration recurses through the plugin path; use
an upstream compatible provider/gateway, or select local providers explicitly.

The local `abstractvoice web` example exposes these smoke-test routes. They are
example routes, not a replacement for AbstractCore Server, and they do not
inherit AbstractCore/Gateway authentication or browser-origin policy:

```bash
abstractvoice web
abstractvoice web --tts-engine openai --stt-engine openai
abstractvoice web --tts-engine openai-compatible --stt-engine openai-compatible --remote-base-url http://localhost:8000/v1
```

- `GET /api/status` -> lightweight server/config status
- `GET /api/voices` -> `VoiceManager.get_profiles()`, `list_available_models()`, `list_cloned_voices()`
- `GET /v1/audio/voices` -> compatible extension for remote profile/voice discovery (`VoiceManager.get_profiles()` + `list_cloned_voices()`)
- `POST /api/tts/provider` (alias: `/api/tts/engine`) -> switch the browser example's base TTS provider through `VoiceManager.set_tts_engine(...)`; resets the base profile/voice to the provider/language default and clears local role clone selections
- `POST /api/voices/select` -> select base TTS, a cloned voice, or a TTS profile; optional local `role="assistant"|"user"` stores browser-example defaults; optional `preload=true` warms a cloned voice by calling a tiny `VoiceManager.speak_to_bytes(...)`
- `POST /api/voices/clone` -> example-only multipart upload for browser voice cloning; stores the uploaded/recorded reference with `VoiceManager.clone_voice(...)` and validates by synthesizing a short sample by default (`validate=false` skips validation)
- `POST /v1/voice/clone` -> compatible extension for remote clone creation; returns `voice_id`/`id` for later `/v1/audio/speech` `voice`
- `POST /api/tts` -> `VoiceManager.speak_to_bytes(...)`; accepts `input`/`text`, `voice`, `role`, `language`, `speed`, `format`/`response_format`, and `sanitize_syntax`
- `POST /api/stt/transcriptions` -> `VoiceManager.transcribe_file(...)`
- `POST /api/stt/transcribe` -> compatibility alias for `/api/stt/transcriptions`
- `GET /api/llm/models` -> example-only model listing for an OpenAI-compatible local provider such as Ollama
- `POST /api/chat` -> example-only non-streaming chat completion proxy; the browser owns history and sends the full short message list
- `POST /v1/audio/speech` and `POST /v1/audio/transcriptions` -> local aliases for quick AbstractCore-compatible smoke tests. In the web example, `voice` may be either a cloned voice id/name or an active-provider profile id.

Local web example payload sketches:

```bash
# Browser-example role default; still resolves to a cloned voice_id.
curl -X POST http://127.0.0.1:5000/api/voices/select \
  -H "Content-Type: application/json" \
  -d '{"role":"assistant","kind":"clone","voice":"my_voice","preload":true}'

# Browser-example cloned voice creation.
curl -X POST http://127.0.0.1:5000/api/voices/clone \
  -F "name=my_voice" \
  -F "provider=omnivoice" \
  -F "reference_text=Exact transcript of the reference audio." \
  -F "file=@reference.wav"

# TTS still maps to VoiceManager.speak_to_bytes(...).
curl -X POST http://127.0.0.1:5000/api/tts \
  -H "Content-Type: application/json" \
  -d '{"input":"Hello.","role":"assistant","response_format":"wav"}' \
  --output hello.wav

# Switch the browser example to Supertonic and its default profile.
curl -X POST http://127.0.0.1:5000/api/tts/provider \
  -H "Content-Type: application/json" \
  -d '{"provider":"supertonic"}'

# Compatible extension: discover profiles/cloned voices from another
# AbstractVoice client configured with remote_base_url=http://127.0.0.1:5000/v1.
curl http://127.0.0.1:5000/v1/audio/voices

# Compatible extension: create a remote cloned voice handle.
curl -X POST http://127.0.0.1:5000/v1/voice/clone \
  -F "name=my_remote_voice" \
  -F "reference_text=Exact transcript of the reference audio." \
  -F "file=@reference.wav"

# Example dialogue call via a local OpenAI-compatible provider.
curl -X POST http://127.0.0.1:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"provider":"ollama","model":"gemma3:1b","messages":[{"role":"user","content":"Say hi in one sentence."}]}'
```

The local `role` field is only a convenience for the example UI. AbstractCore
compatibility remains the capability/plugin contract: clients may pass a
`voice` cloned-voice id to TTS, and AbstractCore routes that to
`VoiceManager.speak_to_bytes(..., voice=...)`.

### AbstractCore tool helpers (manual wiring)

If you prefer to wire tools explicitly, `abstractvoice/integrations/abstractcore.py` provides:

- `make_voice_tools(voice_manager, store) -> list[callable]`
  - Requires `abstractcore` at runtime (it imports `abstractcore.tool`).
  - `store` can be a MediaStore-like object, or an AbstractRuntime-like ArtifactStore (adapted via `RuntimeArtifactStoreAdapter` in `abstractvoice/artifacts.py`).

Tools exposed by `make_voice_tools(...)` (current):
- `voice_tts(text, voice=None, format="wav", run_id=None) -> artifact_ref`
- `voice_profile_list(kind="tts") -> {profiles, active_profile}`
- `voice_profile_set(profile_id, kind="tts") -> {ok, active_profile}`
- `audio_transcribe(audio_artifact|audio_b64, ...) -> {text, transcript_artifact}`

Minimal sketch:

```python
from abstractvoice import VoiceManager
from abstractvoice.integrations.abstractcore import make_voice_tools

vm = VoiceManager(remote_api_key="sk-...")
tools = make_voice_tools(voice_manager=vm, store=artifact_store)
```

Example (provider-agnostic profile selection):

```python
vm = VoiceManager(tts_engine="omnivoice", allow_downloads=False)
vm.set_profile("female_01", kind="tts")
wav_bytes = vm.speak_to_bytes("Hello.", format="wav")
```

TTS metrics (library-level):
- `VoiceManager.speak_to_bytes(...)` / `VoiceManager.speak_to_file(...)` record best-effort stats for the *last* synthesis.
- Call `vm.pop_last_tts_metrics()` to retrieve and clear them (dict with fields like `engine`, `synth_s`, `audio_s`, `rtf`, `sample_rate`).

## Non-contract surface (may change without notice)

- CLI behavior (`abstractvoice/examples/*`)
- Internal adapter details and model catalogs beyond the documented defaults
