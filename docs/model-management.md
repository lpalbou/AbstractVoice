# Model / Voice Management

AbstractVoice core is **Piper-first**:

- **Piper (default)**: small ONNX voices downloaded/managed by `abstractvoice/adapters/tts_piper.py`
- There is **no legacy Coqui model management** in core.
- Heavier engines (torch/transformers) are **opt-in** via extras (e.g. `abstractvoice[chroma]`, `abstractvoice[audiodit]`, `abstractvoice[omnivoice]`).

## What gets downloaded, and when?

Piper voices are stored as small ONNX files under:

- `~/.piper/models`

Downloads are controlled by `allow_downloads`:

- **Library default**: `VoiceManager(..., allow_downloads=True)` may download Piper models on-demand.
- **REPL default**: `python -m abstractvoice cli` runs with `allow_downloads=False` (offline-first), so it will **not** download implicitly.

- Typical voice size: tens of MB per language

## Opt-in engines (HF cache)

Some optional engines download weights via Hugging Face and cache under `~/.cache/huggingface` by default:

- **Chroma cloning**: `python -m abstractvoice download --chroma` (requires `abstractvoice[chroma]`)
- **AudioDiT (LongCat-AudioDiT-1B)**: `python -m abstractvoice download --audiodit` (requires `abstractvoice[audiodit]`)
- **OmniVoice**: `python -m abstractvoice download --omnivoice` (requires `abstractvoice[omnivoice]`)

In offline-first mode (`allow_downloads=False`) these engines will **not** fetch missing weights implicitly.

## Programmatic introspection

- `vm.list_available_models()` returns a dict of known Piper voices and whether they are cached.
- `vm.set_language("<lang>")` loads the voice if cached; it will download only if `allow_downloads=True`.

## CLI

- Use `python -m abstractvoice cli` and `/setvoice` to view available voices (and cache status).
- Prefetch explicitly (offline-first):

```bash
python -m abstractvoice download --piper en
```
