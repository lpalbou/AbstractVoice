# Model / Voice Management

AbstractVoice core is **remote-first**:

- **OpenAI remote audio (default)**: no local model download in the base install.
- **Piper (local)**: small ONNX voices downloaded/managed by `abstractvoice/adapters/tts_piper.py` when you select `tts_engine="piper"`.
- **Supertonic 3 (local)**: recommended fixed-profile ONNX TTS downloaded/managed by `abstractvoice/supertonic/runtime.py` when you select `tts_engine="supertonic"` or prefetch `--supertonic`.
- There is **no legacy Coqui model management** in core.
- Heavier engines (torch/transformers) are **opt-in** via extras (e.g. `abstractvoice[chroma]`, `abstractvoice[audiodit]`, `abstractvoice[omnivoice]`).

## What gets downloaded, and when?

Piper voices are stored as small ONNX files under:

- `~/.piper/models`

(For installs at very deep paths, AbstractVoice may also keep a short symlink under
`~/.cache/abstractvoice/espeak-*` pointing at piper's bundled espeak-ng data — see the Piper
section of `docs/troubleshooting.md`.)

Supertonic 3 artifacts are stored under:

- `~/.cache/abstractvoice/supertonic-3`

Downloads are controlled by `allow_downloads`:

- **Library default**: `VoiceManager()` uses OpenAI remote audio and does not download local models.
- **Local Piper**: `VoiceManager(tts_engine="piper", allow_downloads=True)` may download Piper models on-demand.
- **Local Supertonic**: `VoiceManager(tts_engine="supertonic", allow_downloads=True)` may download Supertonic ONNX artifacts on first synthesis. Profile listing does not download.
- **REPL/web default**: `python -m abstractvoice cli` and `abstractvoice web`
  run with `allow_downloads=False` (offline-first), so they will **not**
  download implicitly. Their interactive TTS `auto` resolver selects installed
  Supertonic first, installed Piper second, then OpenAI remote.

- Typical voice size: tens of MB per language
- Supertonic 3 cache size: roughly 400 MB for the shared ONNX graphs plus all built-in voice styles

## Opt-in engines (HF cache)

Some optional engines download weights via Hugging Face and cache under `~/.cache/huggingface` by default:

- **Chroma cloning**: `python -m abstractvoice download --chroma` (requires `abstractvoice[chroma]`)
- **AudioDiT (LongCat-AudioDiT-1B)**: `python -m abstractvoice download --audiodit` (requires `abstractvoice[audiodit]`)
- **OmniVoice**: `python -m abstractvoice download --omnivoice` (requires `abstractvoice[omnivoice]`; recommended/default local cloning backend)

In offline-first mode (`allow_downloads=False`) these engines will **not** fetch missing weights implicitly.

## Discovery reads the cache, not the engine

Asking which providers and models are available is a filesystem question, and AbstractVoice answers
it that way. A local engine is listed once two things are true:

1. its runtime is installed (the optional extra), and
2. at least one of its models is on this machine.

Nothing about that question loads a model, imports torch, or builds an engine, so provider and model
listing stays fast even when heavy engines are installed. Where presence is read from:

| Engine | Presence is | Location |
| --- | --- | --- |
| `piper` | a downloaded voice pair (`.onnx` + `.onnx.json`) | `~/.piper/models` |
| `supertonic` | the full ONNX + voice-style set | `~/.cache/abstractvoice/supertonic-3` |
| `audiodit` | a cached Hugging Face snapshot holding weights | `~/.cache/huggingface` |
| `omnivoice` | a cached Hugging Face snapshot holding weights | `~/.cache/huggingface` |

A consequence worth knowing: an engine whose extra is installed but whose weights are not downloaded
yet does **not** appear in provider listings. Selecting it still works and still downloads on demand
when `allow_downloads=True` — only discovery is affected. Prefetch it (see above) to have it listed.

If you point an engine at your own checkpoint with `ABSTRACTVOICE_TTS_MODEL` (or `voice_tts_model` in
an integrator config), that id is discoverable too, as long as it is cached or is a local directory
containing weights. It counts only for the engine you selected with `ABSTRACTVOICE_TTS_ENGINE`.

`abstractvoice.local_models` exposes this directly:

```python
from abstractvoice.local_models import cached_tts_model_ids, hf_repo_is_cached

cached_tts_model_ids("piper")       # ['en_US-amy-medium', ...] — only what is downloaded
cached_tts_model_ids("audiodit")    # [] until the weights are prefetched
hf_repo_is_cached("k2-fsa/OmniVoice")
```

## Programmatic introspection

- `vm.list_available_models()` returns a dict of known Piper voices or Supertonic styles, including cache status when the active adapter exposes it.
- `vm.get_profiles()` returns active-engine profiles. Supertonic exposes `M1`-`M5` and `F1`-`F5` without requiring cached weights.
- `vm.set_language("<lang>")` loads the voice if cached; it will download only if `allow_downloads=True` and the selected adapter supports on-demand downloads.
- `vm.set_tts_engine("<engine>")` switches the active base TTS adapter and resets the base profile to that engine/language default. This is the path used by the CLI and web example.

## CLI

- Use `python -m abstractvoice cli` and `/voices models` to view available
  active-engine catalog entries, including Piper voices or Supertonic styles
  with cache status. `/setvoice` still works as a Piper compatibility command.
- Prefetch explicitly (offline-first):

```bash
python -m abstractvoice download --piper en
python -m abstractvoice download --supertonic
```
