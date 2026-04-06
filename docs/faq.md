# FAQ

If you're new: start with `README.md`, then `docs/getting-started.md`.

This FAQ focuses on the questions we expect from first-time users and integrators. For the supported API surface, see `docs/api.md`.

## Installation & compatibility

### What Python versions are supported?

Python `>=3.10` (declared in `pyproject.toml`).

### Do I need system dependencies?

- **TTS (Piper)**: no OS-level speech dependencies (no `espeak-ng`). TTS is implemented by the Piper adapter (`abstractvoice/adapters/tts_piper.py`).
- **Audio I/O (mic/speakers)**: playback and capture use `sounddevice` (PortAudio). Some Linux environments require PortAudio system packages; see `docs/installation.md`.

### I installed the package but `abstractvoice` is “command not found”

- Ensure you installed into the environment you’re using:
  - `python -m pip install abstractvoice`
- From a source checkout, you can always run the REPL via:
  - `python -m abstractvoice cli --verbose`

The console scripts are declared in `pyproject.toml` under `[project.scripts]`.

## Offline-first, downloads, and cache locations

### Why doesn’t the REPL download models automatically?

The REPL intentionally runs in offline-first mode: it constructs `VoiceManager(..., allow_downloads=False)` (see `abstractvoice/examples/cli_repl.py`). This prevents surprise downloads during interactive use.

For library use, `VoiceManager` defaults to `allow_downloads=True` (see `abstractvoice/vm/manager.py`), so it *may* download required assets on demand.

### How do I prefetch models/artifacts explicitly?

Use either:

```bash
abstractvoice-prefetch --stt small
abstractvoice-prefetch --piper en
```

or:

```bash
python -m abstractvoice download --stt small
python -m abstractvoice download --piper en
```

See `docs/installation.md` for the optional cloning/Chroma downloads.

### Where are models and artifacts stored?

Default locations (source of truth in code):

- **Piper voices**: `~/.piper/models` (see `PiperTTSAdapter` in `abstractvoice/adapters/tts_piper.py`).
- **faster-whisper models**: Hugging Face default cache (the adapter uses `download_root=None`; see `abstractvoice/adapters/stt_faster_whisper.py`).
- **OpenF5 artifacts (cloning)**: `~/.cache/abstractvoice/openf5` (see `abstractvoice/cloning/engine_f5.py`).
- **Chroma artifacts (cloning)**: `~/.cache/abstractvoice/chroma` (see `abstractvoice/cloning/engine_chroma.py`).
- **Cloned voice store (your created voices)**: platform app data dir under `abstractvoice/cloned_voices` (see `appdirs.user_data_dir("abstractvoice")` in `abstractvoice/cloning/store.py`).

### How do I delete cached models / reset state?

Stop all running processes, then remove only what you intend to purge:

- Piper voices: `~/.piper/models`
- OpenF5: `~/.cache/abstractvoice/openf5`
- Chroma: `~/.cache/abstractvoice/chroma`
- Cloned voices (metadata + reference bundles): `appdirs.user_data_dir("abstractvoice")/cloned_voices` (see `abstractvoice/cloning/store.py`)

Be aware: the Hugging Face cache is shared with other tools; deleting it can affect other projects.

## REPL usage (voice assistant)

### Can I use the REPL without an LLM server?

Yes. Use `/speak <text>` to test TTS without calling any LLM endpoint (implemented in `abstractvoice/examples/cli_repl.py`). For full end-to-end chat, the REPL calls a configured LLM HTTP endpoint (default is a local Ollama URL; see `docs/repl_guide.md`).

### My LLM prints `<think>...</think>` blocks. Can the REPL hide them?

Yes. The REPL discards `<think>...</think>` blocks from the assistant response before it:

- prints the answer
- stores it in conversation history
- sends it to TTS

This prevents “chain-of-thought” style reasoning from being displayed or spoken.

### How do I enable microphone input?

Mic capture is **off by default**. Enable explicitly:

```bash
abstractvoice --voice-mode stop
```

See `docs/repl_guide.md` for mode descriptions and commands.

### What do the voice modes mean, and which one should I use?

Modes are implemented by wiring TTS playback callbacks to recognizer controls in `abstractvoice/vm/core.py`:

- `stop` (recommended on speakers): keeps listening; during TTS it suppresses normal transcriptions but keeps a stop-phrase detector active so you can say “ok stop” to cut playback.
- `wait` (strict turn-taking): pauses mic processing while speaking.
- `full` (barge-in by speech): allows speech to interrupt TTS; best with AEC or a headset (speakers can self-interrupt).
- `ptt` (push-to-talk): tuned for short utterances; capture is controlled by the integrator/REPL.

Design background: `docs/adr/0002_barge_in_interruption.md`.

### “Ok stop” doesn’t stop playback

Common causes:

- You’re using `wait` mode: mic processing is paused during TTS (so stop phrases can’t be detected while speaking). Use `stop` mode instead (`abstractvoice --voice-mode stop`).
- Your terminal/IDE doesn’t have microphone permission (macOS privacy settings). See `docs/installation.md`.

Stop phrase logic lives in `abstractvoice/recognition.py` and normalization in `abstractvoice/stop_phrase.py`.

## Library integration

### Can I use AbstractVoice in a server/headless environment?

Yes. Prefer the headless-friendly APIs:

- `speak_to_bytes()` / `speak_to_file()` (TTS without speaker playback)
- `transcribe_from_bytes()` / `transcribe_file()` (STT without mic capture)

These are part of the supported contract in `docs/api.md` and implemented in `abstractvoice/vm/tts_mixin.py` and `abstractvoice/vm/stt_mixin.py`.

### How does this integrate with AbstractCore / AbstractRuntime (AbstractFramework)?

AbstractVoice can be used standalone, but it also ships optional integration hooks for the AbstractFramework ecosystem:

- AbstractCore capability plugin entry point: `pyproject.toml` → `[project.entry-points."abstractcore.capabilities_plugins"]`  
  Implementation: `abstractvoice/integrations/abstractcore_plugin.py`
- Tool helpers for AbstractCore: `abstractvoice/integrations/abstractcore.py`
- ArtifactStore adapter (AbstractRuntime-compatible, duck-typed): `abstractvoice/artifacts.py`

See `docs/api.md` (“Integrations”) for the supported surface and code pointers.

### How do I switch language?

Use `VoiceManager(language="...")` at construction time or call `vm.set_language("fr")`.

Language validation is based on the small mapping in `abstractvoice/config/voice_catalog.py` and the default Piper model mapping in `abstractvoice/adapters/tts_piper.py`.

Note: `set_voice(language, voice_id)` exists for backward compatibility, but Piper voice selection is currently best-effort (one default voice per language).

## Voice cloning (optional)

### Is voice cloning included in the base install?

No. Voice cloning is optional:

- Install OpenF5-based cloning: `pip install "abstractvoice[cloning]"`
- Install Chroma runtime deps (GPU-heavy): `pip install "abstractvoice[chroma]"`

Artifacts are still downloaded explicitly via prefetch (see above). User workflow and commands: `docs/repl_guide.md`.

### Why is cloning memory usage so high?

Cloning engines can load multi‑GB model weights (especially Chroma). The code provides best-effort unloading helpers:

- `VoiceManager.unload_cloning_engines(...)` / `unload_piper_voice()` (`abstractvoice/vm/tts_mixin.py`)
- engine `unload()` implementations (`abstractvoice/cloning/engine_f5.py`, `abstractvoice/cloning/engine_chroma.py`)

### Does longer reference audio make cloning slower?

Often yes — *up to an engine-specific cap*.

All cloning engines must decode/resample/encode the reference audio (the “prompt”) before generating speech. More seconds typically means more work. In practice, total latency is usually dominated by the **generation model** (and the requested output length), but the reference length can noticeably affect:

- **First use** (preprocessing + model warmup/compilation)
- **Prompt encoding** (AudioDiT in particular)

Engine-specific behavior in this repo:

- **OpenF5 (`f5_tts`)**:
  - Reference audio is merged → resampled to **24 kHz mono** → clipped to **15s** (`engine_f5.py:_prepare_reference_wav(max_seconds=15.0)`).
  - Longer inputs than 15s won’t increase runtime (they’re truncated).
- **AudioDiT (`audiodit`)**:
  - Prompt audio is concatenated → resampled to the model’s rate (**24 kHz**) → clipped to **15s** (`audiodit/runtime.py:generate_chunks(max_prompt_seconds=15.0)`).
  - Longer prompts can cost more twice: prompt encoding itself, and because prompt frames contribute to the model’s `duration_frames`.
  - Practical starting point: **4–8s** of clean speech is usually the best speed/quality tradeoff.
- **Chroma (`chroma`)**:
  - Prompt audio is normalized to **24 kHz mono PCM16** and clipped to **30s** (`engine_chroma.py:_prepare_prompt_audio_for_processor`).
  - The normalized prompt is cached under `~/.cache/abstractvoice/chroma/prompt_cache`, so the **first** run is slower than subsequent runs with the same prompt file.

Recommendations (general-purpose):

- **Keep prompts short**: start with **~6s** of clean speech (AudioDiT: 4–8s; OpenF5: up to 10–15s is fine; Chroma: 6–15s is typical).
- **Trim silence**: long leading/trailing silence wastes prompt budget and can slow preprocessing.
- **Single speaker, no music**: mixed speakers/background audio reduces identity consistency and can destabilize some models.
- **Use the right transcript**:
  - AudioDiT cloning requires a correct `reference_text` (prompt transcript) matching the prompt audio.
  - For other engines, providing `reference_text` is still often beneficial.

### Why do I see “Output device rejected 24000Hz; using 48000Hz (resampling)”?

Because **24,000 Hz is the model/sample rate**, but your **audio output device** (speakers/headphones) often only supports **44.1 kHz or 48 kHz** for live playback streams.

What this means in practice:

- **Cloning + synthesis run at ~24 kHz internally** (AudioDiT/OpenF5/Chroma prompts and outputs are normalized around 24 kHz in this repo).
- **Playback may use 48 kHz** because PortAudio/CoreAudio rejects a 24 kHz output stream for many macOS devices (built‑in speakers, Bluetooth, HDMI, etc.).
- AbstractVoice then **resamples for playback** so you can still hear the 24 kHz model output reliably.

This warning is about **speaker device constraints**, not about what sample rate the cloning model uses.

Note: AbstractVoice tries to open the playback stream at the audio’s natural rate first (e.g. 24 kHz for AudioDiT). If that fails, it falls back to the device’s default (often 48 kHz) and resamples.

## Licensing

### Is AbstractVoice MIT licensed? What about the voices/models?

The **library code** is MIT licensed (`LICENSE`). **Model weights and voice files are separate assets** with their own licenses and restrictions.

Read `docs/voices-and-licenses.md` before distributing voices/models.
