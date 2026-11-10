## Task 0070: Qwen3-TTS on CUDA is launch-bound (GPU idle 90% of a generation)

**Date**: 2026-11-10  
**Status**: Completed (2026-11-10, commit fbc04cd)  
**Priority**: P1

---

## Main goal

Qwen3-TTS on the PyTorch/CUDA backend speaks faster than real time on a mid-range NVIDIA card, as
the MLX backend does on a Mac laptop.

## Context / problem statement

Measured 2026-11-10 on a Linux test server (Ubuntu 24.04, Quadro RTX 5000 16 GB, compute
capability 7.5, 4 vCPU, torch 2.15.0+cu132, transformers 5.20.2), `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`,
model resident and warm, 9 s of speech:

| dtype | total | rtf | GPU busy |
|---|---|---|---|
| bfloat16 (old policy default, emulated on Turing) | 19.7 s | 2.14 | ~10% |
| float32 | 14.5 s | 1.64 | ~10% |
| processor only (4 vCPU) | 35 s | 3.95 | — |

Breakdown (float32): the code predictor (`predictor.py` `generate_codebooks`, one call per frame,
15 steps over a 5-layer model with a fresh `DynamicCache`) was 71% of wall clock at ~8 ms per
step; the talker step through Transformers `generate` 28% at 37-47 ms per frame; codec decode 1%.
torch.profiler: Self CUDA time 1.4 s against Self CPU 6.4 s. Kernel launches and Python, not compute.

## Decision

The predictor's fixed-depth frame is captured once as a CUDA graph (per input shape and sampling
setting) and replayed per frame; the capture records the eager loop's own kernels, so logits, IDs
and Philox RNG consumption are unchanged. The draw inside the graph is the exponential race
(`torch.multinomial`'s own n=1 construction) without its two host syncs per step; probabilities are
validated once per frame before any ID escapes. CPU and MPS keep the eager loop; hooks on the model,
a weights move or dtype change (recaptured), a nested capture or a failed capture fall back to eager
with one `#FALLBACK` line; `predictor=reference` untouched; MLX untouched. Cyclic GC is paused during
capture (another model's graphs being collected mid-capture broke it 3 of 3 times).

The talker step (now 57% of wall clock at 36 ms per frame) was left as is: a static cache there
changes its numerics; next gain if needed.

## Report

Quadro RTX 5000, float32, 0.6B CustomVoice, ~9 s of speech:

| | before | after |
|---|---|---|
| rtf, warm takes | 1.60-1.72 | 0.79-0.83 |
| predictor per frame | 89 ms | 25.8 ms |
| mean GPU utilization | 28% | 54% |
| first take after load | ~1.0 | ~1.01 (includes the 0.2-0.4 s capture) |

Parity, on the server: unit tests on CUDA across float32/bfloat16, batch 1 and 3, four sampling
settings, both samplers, capturing call and replays: codes, every step's logits and the RNG state
bitwise equal to `predictor.generate` (126 passed, three runs). Opt-in real-checkpoint test
(`tests/test_qwen3_tts_predictor_e2e_optional.py`, now also checking the graph's logits) passes for
0.6B CustomVoice and 0.6B Base; whole-utterance audio bitwise equal across the six predictor modes
on both checkpoints; graph vs eager audio with the same seed: max difference 0.0. Mac CI selection
(MLX backend): unchanged, CUDA-only tests skipped. Measurement scripts: `untracked/cuda-graphs/`.

## Follow-ups

- 0071 (CUDA out of memory has no fallback), 0068 (parity test on Apple silicon).
