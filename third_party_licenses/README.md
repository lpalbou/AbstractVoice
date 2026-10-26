# Third-party licenses

This folder contains license texts for third-party **code vendored into this repository**.

- `longcat_audiodit_license.txt`: LongCat-AudioDiT upstream license (MIT). Applies to the derived implementation under `abstractvoice/audiodit/*`.
- `qwen_asr_license.txt`: Apache 2.0 license for the derived Qwen3-ASR implementation under `abstractvoice/qwen3_asr/*` (sourced from upstream `qwen-asr`).
- `supertone_supertonic_notice.txt`: notice for optional Supertonic 3 model artifacts downloaded from Hugging Face. The model files are not vendored into this repo.
- `qwen3_tts_notice.txt`: notice for the derived Qwen3-TTS PyTorch implementation under `abstractvoice/qwen3_tts/*` (from the Apache-2.0 `qwen-tts` package).
- `mlx_audio_notice.txt`: MIT license and attribution for the MLX port under `abstractvoice/qwen3_tts/mlx/*`, derived from mlx-audio's Qwen3-TTS implementation.
- `omnivoice_notice.txt`: notice for optional OmniVoice package/model artifacts downloaded from upstream sources. The model files are not vendored into this repo.

Notes:
- Runtime dependencies installed via pip are not vendored here; see `ACKNOWLEDGMENTS.md` and `docs/dependencies.md`.
- Model weights and voice files are not shipped with this repo; they have separate licenses (see `docs/voices-and-licenses.md`).
