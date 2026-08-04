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

## Supertonic 3 fixed-profile TTS

Supertonic is optional local TTS (`abstractvoice[supertonic]`,
`abstractvoice[apple]`, or `abstractvoice[gpu]`). It is not a voice-cloning
engine. AbstractVoice exposes the built-in style profiles `M1`-`M5` and
`F1`-`F5` through the standard `VoiceProfile` API.

- Model source: `https://huggingface.co/Supertone/supertonic-3`
- License file: `https://huggingface.co/Supertone/supertonic-3/blob/main/LICENSE`
- Default cache: `~/.cache/abstractvoice/supertonic-3`
- Local notice: `third_party_licenses/supertone_supertonic_notice.txt`

Prefetch explicitly for offline-first use:

```bash
python -m abstractvoice download --supertonic
```

The AbstractVoice adapter/runtime code is internal and does not depend on
Supertone's external Python SDK. The downloaded model weights and style files
remain third-party assets with their own license terms.

## Voice cloning (optional)

AbstractVoice supports voice cloning behind optional extras:

- `abstractvoice[cloning]` (OpenF5-based; large artifacts)
- `abstractvoice[chroma]` (Chroma-4B; very large; GPU-heavy)
- `abstractvoice[audiodit]` (LongCat-AudioDiT-1B; large weights via HF)
- `abstractvoice[omnivoice]` (OmniVoice; recommended/default local cloning backend; very large; torch/transformers)
- `abstractvoice[qwen3-tts]` (Qwen3-TTS Base checkpoints; ~2.5-4.5 GB per snapshot via HF; torch/transformers)

Licensing is engine- and model-dependent; verify:

- the Python package license (code)
- any model weights / checkpoints you download
- any dataset-specific restrictions

Note on vendored code:

- This repo includes a derived implementation of LongCat-AudioDiT under `abstractvoice/audiodit/*` to avoid `trust_remote_code`.
- Upstream license text is included in `third_party_licenses/longcat_audiodit_license.txt`.
- This repo includes a derived implementation of Qwen3-ASR under `abstractvoice/qwen3_asr/*` so `Qwen/Qwen3-ASR-1.7B` can run without `trust_remote_code`.
- Upstream license text is included in `third_party_licenses/qwen_asr_license.txt`.
- This repo includes a derived implementation of Qwen3-TTS under `abstractvoice/qwen3_tts/*` (from the Apache-2.0 `qwen-tts` package) so the `Qwen/Qwen3-TTS-12Hz-*` checkpoints can run without `trust_remote_code`.
- Notice and upstream pointers are included in `third_party_licenses/qwen3_tts_notice.txt`.

For AudioDiT specifically, verify the model card/source for:

```bash
python -m abstractvoice download --audiodit
```

For OmniVoice specifically, verify the model card/source for:

```bash
python -m abstractvoice download --omnivoice
```

Local notices:

- Supertonic notice: `third_party_licenses/supertone_supertonic_notice.txt`
- OmniVoice notice: `third_party_licenses/omnivoice_notice.txt`

## Practical guidance

- Treat **model weights** as third-party assets with their own compliance requirements.
- Pin exact versions/revisions in production and record them for audits.
- If you distribute models with your product, include the required notices.
