## Task 0068: Qwen3-TTS predictor parity test on Apple silicon

**Date**: 2026-11-09  
**Status**: Proposed  
**Priority**: P2

---

## Main goal

`tests/test_qwen3_tts_predictor_e2e_optional.py` (real checkpoints, opt-in with
`ABSTRACTVOICE_RUN_QWEN3_TTS_TESTS=1`) gives a trustworthy answer on a Mac: it either checks the
PyTorch fast predictor against Transformers' reference or says why it cannot, instead of failing
for reasons unrelated to the predictor.

## Context / problem statement

Observed during the 0.21.0 release check (2026-11-09, Apple M-series, transformers 5.20.2,
torch 2.12.0, all five checkpoints cached):

1. With the default backend policy (ADR 0008: MLX on Apple silicon) all five parametrizations
   fail with `AttributeError: 'NoneType' object has no attribute 'model'` because the test reaches
   into `runtime._model.model.talker.code_predictor`, which only exists on the PyTorch backend.
   The test predates the MLX backend and never selects a backend.
2. With the PyTorch backend forced, the per-frame checks (sequences, RNG state, 15 logits) pass
   for every checkpoint, but the whole-utterance parity (`np.testing.assert_array_equal(audio,
   expected_audio)` across predictor modes with the same seed) fails for
   `Qwen/Qwen3-TTS-12Hz-0.6B-Base` with audio of different lengths on each run. The same test on the
   v0.20.1 tree fails identically, so this is not a 0.21.0 regression. The 1.7B VoiceDesign case
   is intermittent on both trees: 2 of 2 runs failed on 0.21.0, 1 of 3 failed on the v0.20.1 tree,
   always with audio of different lengths across modes (for example 55680 vs 88320 samples).
3. CI never runs this file (its `model_download` marker is deselected), so these gaps are only
   seen by hand.

## Constraints

- No behaviour change to the predictor or the runtime without a measured reason; this is a test
  faithfulness item first.
- Keep the opt-in nature: real checkpoints, no downloads in CI.

## Research & references

- ADR 0008 (Qwen3-TTS backends).
- `abstractvoice/qwen3_tts/predictor.py` (`generate_codebooks`, `_unsupported_reason`), unchanged
  since 533aea3 (Transformers 5.9+).
- Release check logs, 2026-11-09 (core seat scratch; shapes 65280 vs 69120 on 0.21.0, 59520 vs
  57600 on v0.20.1 for the 0.6B Base case).

## Decision

Proposed: (a) the test requests `backend="torch"` explicitly and skips with a reason when PyTorch
cannot run; (b) separate the per-frame parity (exact, keeps the hard assertion) from the
whole-utterance parity, and make the latter report its measurement instead of asserting exact
equality until the source of the run-to-run length difference on MPS is understood (sampling RNG
consumption in Transformers 5.20 `generate` is the first suspect); (c) if the difference is a real
divergence of the fast predictor on current Transformers, fix the predictor and restore the hard
assertion.

## Dependencies

- None blocking. Related: 0946 (Qwen3-ASR Transformers floor), the Qwen3-TTS CI drift jobs.

## Implementation plan

1. Pass `backend="torch"` in the test's runtime constructions; `pytest.skip` when
   `select_backend("torch")` raises.
2. Split the whole-utterance block into a measured report (lengths, RTF) plus an exactness check
   that is `xfail(strict=False)` with a reason naming this item, on MPS only.
3. Reproduce the length difference with a 20-line script outside pytest; bisect Transformers
   versions (5.9.0 floor vs 5.20.2).
4. Depending on 3: fix or document, then remove the xfail.

## Success criteria

- On a Mac with cached checkpoints and the flag set, the file passes or skips with a stated reason
  on both backend policies; no `AttributeError`.
- The exactness check is a hard assertion again, or a documented known limit with a measurement.

## Test plan

- `ABSTRACTVOICE_RUN_QWEN3_TTS_TESTS=1 pytest -o addopts="" tests/test_qwen3_tts_predictor_e2e_optional.py`
  with and without a forced PyTorch backend; the Linux CPU CI job for Qwen3-TTS stays green.
