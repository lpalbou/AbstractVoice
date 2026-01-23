# Model / Voice Management

AbstractVoice core is **Piper-first**:

- **Piper (default)**: small ONNX voices downloaded/managed by `abstractvoice/adapters/tts_piper.py`
- There is **no legacy Coqui model management** in core.

## What gets downloaded, and when?

Piper voices are downloaded **on-demand** the first time you use a language (via `VoiceManager.set_language()` or `VoiceManager.speak_to_*()`).

- Default cache dir: `~/.piper/models`
- Typical voice size: tens of MB per language

## Programmatic introspection

- `vm.list_available_models()` returns a dict of known Piper voices and whether they are cached.
- `vm.set_language("<lang>")` will download the Piper voice for that language if needed.

## CLI

- Use `abstractvoice cli` and `/setvoice` to view available voices (and cache status).

