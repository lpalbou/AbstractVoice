# Troubleshooting

Symptom-oriented fixes for the problems users hit most often. Each entry names what you see, why it
happens, and what to do.

Related pages:

- `docs/installation.md` — install commands, extras, and audio device setup per platform
- `docs/known-issues.md` — tracked defects and engine caveats in this release
- `docs/faq.md` — conceptual questions and offline-first behavior
- `docs/model-management.md` — cache locations and what gets downloaded when

## Installation And Imports

### `abstractvoice: command not found`

The console script is installed into the environment's `bin`/`Scripts` directory. If you installed
with `pip install --user` or a different interpreter, that directory may not be on `PATH`. Run the
module form instead, which always works from the interpreter that has the package:

```bash
python -m abstractvoice cli
```

### An engine raises `requires optional dependencies`

Local engines ship behind extras so the base install stays remote-only. Install the matching extra:

```bash
pip install "abstractvoice[piper]"
pip install "abstractvoice[supertonic]"
pip install "abstractvoice[audiodit]"
pip install "abstractvoice[omnivoice]"
```

Profile bundles install several at once: `abstractvoice[all-apple]`, `abstractvoice[all-gpu]`.

### `OpenAI audio requires OPENAI_API_KEY or remote_api_key=...`

The library default is remote-first. Either export a key, point at an OpenAI-compatible server, or
select a local engine explicitly:

```bash
export OPENAI_API_KEY=sk-...
# or
export OPENAI_BASE_URL=http://localhost:8000/v1
# or
python -m abstractvoice cli --tts-engine supertonic
```

## Provider And Model Discovery

### A local engine is installed but not listed

Discovery lists a local engine when its runtime is installed **and** at least one of its models is on
this machine. Installing the extra alone is not enough. Prefetch the engine:

```bash
python -m abstractvoice download --audiodit
python -m abstractvoice download --omnivoice
python -m abstractvoice download --supertonic
python -m abstractvoice download --piper en
```

Then confirm what the cache actually holds:

```python
from abstractvoice.local_models import cached_tts_model_ids

cached_tts_model_ids("audiodit")
cached_tts_model_ids("piper")
```

An empty list after a successful prefetch usually means the download landed in a different cache than
the one being read. Discovery reads the standard Hugging Face cache, so check `HF_HOME` and
`HF_HUB_CACHE` are the same for the prefetch and for the process doing the listing. A partially
downloaded snapshot also reads as absent on purpose: a snapshot counts only when it holds weights.

Selecting an unlisted engine still works and still downloads on demand when `allow_downloads=True`.
Only the listing is affected.

### My own checkpoint is not selectable

A checkpoint set through `ABSTRACTVOICE_TTS_MODEL` counts for the engine you selected with
`ABSTRACTVOICE_TTS_ENGINE`, and only when it is cached or is a local directory containing weights.
Set both:

```bash
export ABSTRACTVOICE_TTS_ENGINE=audiodit
export ABSTRACTVOICE_TTS_MODEL=myorg/my-audiodit-finetune
```

Integrators passing a config dict use `voice_tts_engine` and `voice_tts_model` for the same effect.

### Listing pauses for about five seconds

A configured remote provider is unreachable. All remote providers are probed at once under a single
5-second budget, so this is one pause for the whole listing rather than one per provider. Lower it
with `ABSTRACTVOICE_DISCOVERY_TIMEOUT_S`, or unset `OPENAI_BASE_URL` if it points somewhere that is
no longer running.

### A provider is listed with no models or voices

Check whether it was reachable. `voice_catalog()` marks a provider it could not reach in time rather
than reporting it as empty:

```python
catalog = core.voice.voice_catalog()
catalog["unreachable_tts_providers"]
catalog["tts_catalog_by_provider"]["openai"].get("unreachable")
```

If the provider is flagged, the empty lists mean "not reached", not "nothing offered". Raise
`ABSTRACTVOICE_DISCOVERY_TIMEOUT_S` for a slow server, or fix the endpoint. If it is not flagged, the
provider genuinely reported nothing — check the model ids configured for it.

## Piper

### Synthesis dies with `Error processing file '.../espeak-ng-data/phontab': No such file or directory`

The path in the message points at a directory that has never existed on your machine (often under
`/Users/runner/work/piper1-gpl/...`). The cause is an install-path length limit: espeak-ng, which
piper's wheels bundle for phonemization, stores its data directory in a fixed 160-character buffer
on macOS and Linux (230 on Windows). When your environment is installed at a deep path — containers,
monorepos, Nix stores — the bundled `piper/espeak-ng-data` directory can exceed that limit, and
espeak-ng falls back to the path compiled in on piper's build machine, then exits the process when
it is missing.

AbstractVoice handles this automatically: when the bundled directory is over the limit, synthesis
routes espeak-ng through a short symlink under `~/.cache/abstractvoice/`, and if no short alias can
be created it raises a clear `RuntimeError` instead of letting the process die. If you see this
error anyway, you are on an AbstractVoice older than 0.10.20 or driving `piper` directly; either
upgrade, install the environment at a shorter path, or pass a short
`PiperVoice.load(..., espeak_data_dir=...)` alias yourself.

Check your install's path length:

```python
from piper.phonemize_espeak import ESPEAK_DATA_DIR
print(len(str(ESPEAK_DATA_DIR)))  # must be under 160 (macOS/Linux) for espeak-ng itself
```

## Audio Devices

### No sound, or `PortAudioError` on start

Platform setup differs; see the audio device section of `docs/installation.md` for macOS, Linux, and
Windows. Common causes are a missing PortAudio system package on Linux, an unset default output
device, and microphone permission not yet granted on macOS.

### Speech is recognized while the assistant is speaking

That is echo, and voice modes control it. See the voice modes entry in `docs/faq.md` and the
coordination section of `docs/architecture.md`.

## Performance

### First synthesis is much slower than later ones

The first call loads the model and opens the audio device. Prefetch weights ahead of time, and
`AdapterTTSEngine.warmup_audio_output()` opens the playback stream before the first utterance.

### A local engine loads weights when you only wanted to browse

Reach for the discovery calls, which never load an engine: `available_providers()`,
`list_models(...)`, `list_tts_voices(provider=...)`, and `voice_catalog(providers_only=True)` or
`voice_catalog(provider=<local>)`. The unfiltered `voice_catalog()`, `list_profiles()`, and
`list_cloned_voices()` report the active engine's live state and so build it. See the cost model in
`docs/api.md`.

## Still Stuck

Open an issue with the platform, Python version, install extras, the exact command, and the full
traceback: `.github/ISSUE_TEMPLATE/bug_report.yml`. For security reports, follow `SECURITY.md`.
