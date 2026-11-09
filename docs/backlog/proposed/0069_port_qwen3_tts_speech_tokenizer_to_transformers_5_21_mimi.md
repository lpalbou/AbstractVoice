## Task 0069: Port the Qwen3-TTS speech tokenizer to Transformers 5.21's Mimi layout

**Date**: 2026-11-09  
**Status**: Proposed  
**Priority**: P1

---

## Main goal

Qwen3-TTS runs on the PyTorch backend with current Transformers (5.21 and later), so the
`transformers<5.21` pin on the platform profiles and `abstractvoice[qwen3-tts]` can be lifted.

## Context / problem statement

Transformers 5.21 (2026-10-26) split `MimiModel` into `MimiEncoderModel` / `MimiDecoderModel`
sub-models (`AutoModel.from_config(config.encoder_config / decoder_config)`) and made
`MimiModel.upsample` and `MimiModel.decoder_transformer` read-only, deprecated properties that
delegate to `self.decoder`. The speech tokenizer AbstractVoice ships
(`abstractvoice/qwen3_tts/modeling_qwen3_tts_tokenizer_v2.py`) subclasses `MimiModel` for its
encoder and frees the decoder half by assigning `None` to `upsample`, `decoder_transformer` and
`decoder` in `__init__`; on 5.21+ that assignment raises
`AttributeError: property 'upsample' of 'Qwen3TTSTokenizerV2Encoder' object has no setter`, so no
Qwen3-TTS checkpoint loads. Found 2026-11-09 on a clean Ubuntu 24.04 NVIDIA host (transformers
5.22.2, then 5.21.1); 5.20.2 works. The CI "transformers latest" job did not catch it because no
test constructed the codec; `tests/test_qwen3_tts_codec_constructs.py` now does (meta device, from
the checkpoint configuration copied under `tests/data/`).

0.21.0 ships the pin `transformers>=5.4.0,<5.21` (profiles) / `>=5.9.0,<5.21` (`qwen3-tts`).

## Constraints

- Identical audio before and after the port on the five Qwen3-TTS checkpoints (codec round trip,
  speaker similarity of a clone); weights load with no missing/unexpected keys on 5.21+.
- 5.9.0 (the declared floor) keeps working, or the floor moves with a changelog note.
- No behaviour change outside the tokenizer module.

## Research & references

- transformers 5.21 `models/mimi/modeling_mimi.py`: `MimiModel.__init__` builds
  `self.encoder = AutoModel.from_config(config.encoder_config)` and
  `self.decoder = ... decoder_config`; `_keys_to_ignore_on_load_unexpected` on the encoder model
  lists `^decoder\.`, `^decoder_transformer\.`, `^upsample\.`, `^quantizer\.`.
- The vendored encoder's state-dict key layout must map onto the new sub-model layout (a
  conversion map, or keep the flat layout by not subclassing `MimiModel` and composing the
  encoder pieces directly).

## Decision

Proposed: stop subclassing `MimiModel`. Compose the encoder from the Mimi building blocks the
tokenizer actually uses (encoder conv stack, encoder transformer, downsample, quantizer), load the
checkpoint's flat keys directly, and drop the `None` assignments. This removes the dependency on
`MimiModel`'s internal layout altogether, so later Mimi refactors cannot break the load.

## Dependencies

- None blocking. Related: 0068 (predictor parity test), the Qwen3-TTS CI drift jobs.

## Implementation plan

1. Write the composed encoder with the same module names as the checkpoint keys.
2. Run `tests/test_qwen3_tts_codec_constructs.py` against 5.9.0, 5.20.2 and the newest release.
3. Real-weights parity on the five checkpoints (codec encode/decode round trip, one clone take)
   against 5.20.2 output, PyTorch backend forced.
4. Lift the pin, update `docs/installation.md` (the Qwen3-TTS note), changelog.

## Success criteria

- Fresh `pip install "abstractvoice[gpu]"` on Linux with the newest Transformers loads and speaks
  with Qwen3-TTS; CI "latest" job green with the pin removed.

## Test plan

- The construction test above; the opt-in real-weights e2e files with the PyTorch backend forced
  (see 0068); the Linux GPU host check (`gpu_check.py` recipe in the 2026-11-09 release notes).
