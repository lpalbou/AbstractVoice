# Voice models and licensing (important)

AbstractVoice (this library) is MIT-licensed, but **voice models / weights are licensed separately**.

This document is intentionally conservative: it does not attempt to “summarize” every upstream license.
For any serious use (especially commercial), verify license terms at the **source** of the model you download.

## Piper voices (default TTS)

AbstractVoice uses Piper for cross-platform, dependency-light TTS.

- Piper voice weights are sourced from the upstream Piper voice catalog (commonly distributed via `rhasspy/piper-voices`).
- **Licenses vary per voice** (dataset + model card).
- Cache directory (by default): `~/.piper/models`

Recommended verification workflow:

1) Download the voice you intend to ship/use:

```bash
python -m abstractvoice download --piper en
```

2) Record which voice/model files you distribute (filenames under `~/.piper/models`).
3) Verify the upstream license/model card for that exact voice.

## STT models (faster-whisper)

STT weights are fetched through the Faster-Whisper / HuggingFace caching mechanisms.
Licensing is model-dependent (e.g. Whisper variants).

For offline deployments, prefetch explicitly:

```bash
python -m abstractvoice download --stt small
```

## Voice cloning (optional)

AbstractVoice supports voice cloning behind optional extras:

- `abstractvoice[cloning]` (OpenF5-based; large artifacts)
- `abstractvoice[chroma]` (Chroma-4B; very large; GPU-heavy)

Licensing is engine- and model-dependent; verify:

- the Python package license (code)
- any model weights / checkpoints you download
- any dataset-specific restrictions

## Practical guidance

- Treat **model weights** as third-party assets with their own compliance requirements.
- Pin exact versions/revisions in production and record them for audits.
- If you distribute models with your product, include the required notices.

