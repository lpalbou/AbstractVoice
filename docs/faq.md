# FAQ

If you're new: start with `README.md`, then `docs/getting-started.md`.

This FAQ focuses on the questions we expect from first-time users and integrators. For the supported API surface, see `docs/api.md`.

## Installation & compatibility

### What Python versions are supported?

Python `>=3.8` (declared in `pyproject.toml`).

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

## Licensing

### Is AbstractVoice MIT licensed? What about the voices/models?

The **library code** is MIT licensed (`LICENSE`). **Model weights and voice files are separate assets** with their own licenses and restrictions.

Read `docs/voices-and-licenses.md` before distributing voices/models.
