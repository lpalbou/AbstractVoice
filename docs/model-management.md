# Model / Voice Management

AbstractVoice core is **Piper-first**:

- **Piper (default)**: small ONNX voices downloaded/managed by `abstractvoice/adapters/tts_piper.py`
- There is **no legacy Coqui model management** in core.

## What gets downloaded, and when?

Piper voices are stored as small ONNX files under:

- `~/.piper/models`

Downloads are controlled by `allow_downloads`:

- **Library default**: `VoiceManager(..., allow_downloads=True)` may download Piper models on-demand.
- **REPL default**: `python -m abstractvoice cli` runs with `allow_downloads=False` (offline-first), so it will **not** download implicitly.

- Typical voice size: tens of MB per language

## Programmatic introspection

- `vm.list_available_models()` returns a dict of known Piper voices and whether they are cached.
- `vm.set_language("<lang>")` loads the voice if cached; it will download only if `allow_downloads=True`.

## CLI

- Use `python -m abstractvoice cli` and `/setvoice` to view available voices (and cache status).
- Prefetch explicitly (offline-first):

```bash
python -m abstractvoice download --piper en
```
