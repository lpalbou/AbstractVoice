## Task 0061: Qwen3-TTS on MLX (Apple silicon backend with native frame streaming)

**Date**: 2026-10-26
**Status**: Completed
**Priority**: P0
**Completed**: 2026-10-26

---

## Main goal(s)

- Run the three Qwen3-TTS modes (CustomVoice, VoiceDesign, Base cloning) on the Apple GPU through
  MLX, behind the existing public surfaces, fast enough for conversation: 1.7B CustomVoice at a
  real-time factor of 0.7 or better with first streamed audio within 1.5 s; 0.6B at 0.4 / 0.6 s.
- Stream codec frames of ONE generation per reply, so streamed speech starts early and the voice
  does not drift between sentences.

## Secondary goal(s)

- Keep the PyTorch path (CUDA, CPU) behaviourally unchanged.
- Load the original Qwen bf16 safetensors from the snapshot already in the Hugging Face cache; no
  second download, no third-party conversion repository.
- Low-level runtime dependencies only (`mlx`, `tokenizers`); no `torch`/`transformers` import on
  the MLX path.

## Context / problem statement

On Apple silicon the vendored PyTorch Qwen3-TTS is kernel-launch bound: eager MPS pays a fixed
cost per kernel and each 80 ms frame needs one talker step plus fifteen code-predictor steps.
Measured on the owner's laptop (REPORT 2026-10-26, section 7): warm RTF 0.9-1.2 on a 63-word
reply, first streamed audio 9 s, the 0.6B no faster than the 1.7B. Streamed delivery also
synthesized sentence-sized segments as independent samples (temperature 0.9, top-k 50), so the
voice drifted between sentences. mlx-audio's MLX port of the same checkpoints ran the 1.7B at RTF
0.61 with first audio in 1.55 s on the same machine.

## Constraints

- Public API of `Qwen3TTSRuntime`, `Qwen3TTSAdapter`, `Qwen3TTSVoiceCloningEngine`, CLI/REPL
  flags, plugin settings and the web example unchanged for callers; `runtime_info()["backend"]`
  added.
- The PyTorch-specific `predictor` / `sampler` switches must be accepted and reported.
- The package must still import on Python 3.9 (`from __future__ import annotations`, no
  runtime-evaluated `X | None`); everything MLX-specific must skip cleanly on Linux CI.
- Vendored model code with licence notices; permissive licences only.

## Research & references

- mlx-audio 0.6.1 `qwen3_tts` (MIT): the starting point for the port. Findings that changed the
  port: its ICL prompt layout differs from upstream's default (`non_streaming_mode=False` for
  clones), its repetition penalty uses a 64-token window where Hugging Face's generation stack
  penalizes the whole generated history, it skips `min_new_tokens=2`, its codec decoder attends
  over the full history where upstream uses a 72-frame sliding window, and its talker MRoPE
  angles come from an fp32 matmul that drifts at long positions (the three position streams are
  identical for TTS, so plain `mx.fast.rope` is exact).
- Vendored PyTorch model (`modeling_qwen3_tts.generate`, `generate_icl_prompt`,
  `orchestration.py`) as the contract: prompt layouts, sampling order, speaker/dialect rules.
- Hugging Face `Qwen2Converter`: the fast-tokenizer recipe reproduced with `tokenizers` from the
  snapshot's `vocab.json`/`merges.txt` (the snapshots ship no `tokenizer.json`).
- Hugging Face `MimiModel` (encoder layout, sliding window 250, exact GELU) for the codec encoder.
- Benchmark/parity harness and optimisation study by the parallel worker (per-frame breakdown:
  the code predictor's 15 sequential steps dominate; fused projections and a compiled predictor
  loop gain ~20%, pipelining is noise-level on a loaded machine).

## Decision

- Two backends behind one runtime, chosen by a framework-free policy (ADR 0008): `mlx` on
  macOS/arm64 with `mlx` importable, else `torch`; `ABSTRACTVOICE_QWEN3_TTS_BACKEND` and
  `Qwen3TTSSettings(backend=...)` override; an impossible `mlx` request is a plain error.
- Vendor and re-align mlx-audio's port on the PyTorch contract rather than depend on `mlx-audio`
  (it pulls `scipy`, `sentencepiece`, `sounddevice`, pins `mlx==`, and its Qwen3-TTS diverges from
  upstream in the points above).
- Strict named weight loading from the original safetensors (every tensor must land on a
  same-shape parameter; key drift is a hard error), 8-bit talker quantization on load (bf16/4-bit
  selectable). No converted-weight cache: load is well under the priming time.
- Frame-level streaming from one stateful codec decode; a 6-frame first chunk then 12-frame
  chunks; fades only at utterance edges (the runtime withholds the last 5 ms of each chunk so the
  final fade-out needs no look-ahead). The adapter exposes `native_frame_streaming`; the mixin and
  the cloning engine pass a whole reply (up to 800 characters) as one segment and skip per-chunk
  silence trimming for native streams.
- Priming generation at load (greedy, a few frames, plus a reference encode on Base checkpoints).

## Dependencies

- ADR 0005 (device/dtype inside the PyTorch backend), ADR 0008 (this backend policy).
- Backlog 0950 (predictor/sampler surfaces) for the switches the MLX backend reports as not applied.

## Implementation plan

1. Vendored MLX package `abstractvoice/qwen3_tts/mlx/` (config, cache, talker, codec, speaker
   encoder, tokenizer, weights, model).
2. Backend policy module and runtime dispatch; `stream_*` generators; priming; `runtime_info`.
3. Adapter / mixin / cloning-engine seams for native frame streaming; `engine_runtime` follows
   the policy.
4. Packaging (`mlx`, `tokenizers` with darwin/arm64 markers), licence notice, acknowledgments.
5. Tests: policy + seams (torch/mlx-free), tokenizer identity, gated real-weights e2e.
6. Docs: README, api, installation, dependencies, troubleshooting, architecture, CHANGELOG,
   llms, ADR 0008, this item.

## Success criteria

- Acceptance numbers above on the owner's machine; parity (WER) comparable with PyTorch;
  VoiceDesign and clone work; speaker presets identical; multi-sentence reply is one generation
  with contiguous chunks; default suite green; MLX-specific tests skip on Linux.

## Test plan

- `python -m pytest -q` (default suite).
- `python -m pytest -q tests/test_qwen3_tts_backend_selection.py tests/test_qwen3_tts_mlx_tokenizer.py`.
- `ABSTRACTVOICE_RUN_QWEN3_TTS_MLX_TESTS=1 python -m pytest -q -m model_download tests/test_qwen3_tts_mlx_e2e_optional.py`.
- Parity probes against the PyTorch model (prompt embeddings, prefill logits, decoder waveform,
  speaker embedding, reference codes) and Whisper transcription of both backends' output.

---

## Report

### Summary

- Shipped the MLX backend behind the existing surfaces; the PyTorch path is unchanged.
- Measured on the owner's Apple silicon laptop (18 cores, macOS, MLX 0.31.2, Python 3.12) on
  2026-10-26 with the parallel worker's harness (`bench_qwen3_mlx.py`, standard 63-word text,
  speaker `ryan`, English, upstream sampling 0.9/50/1.0/1.05, 3 warm runs, 8-bit talker, first
  chunk 6 frames then 12). **Other work was running throughout** (1-minute load average between
  30 and 320 on 18 cores), so these are upper bounds, not quiet-machine figures:

  | Checkpoint | Long reply RTF (min / median) | First streamed audio | Chunks / stalls | Load incl. prime | Peak RSS / MLX peak |
  |---|---|---|---|---|---|
  | 1.7B CustomVoice, 8-bit | 0.52 / 0.54 (target 0.7) | 0.26-0.32 s (target 1.5 s) | 29 x 0.96 s, 0 stalls | 2.1 s (0.36 s load + 0.12 s prime when the file pages are cached) | 4.4 GB / 4.0 GB |
  | 0.6B CustomVoice, 8-bit | 0.25 / 0.29 (target 0.4) | 0.15 s (target 0.6 s) | 29 x 0.96 s, 0 stalls | 1.3 s (0.23 s + 0.11 s) | 2.4 GB / 2.8 GB |
  | 1.7B CustomVoice, bf16 (for reference) | 0.92 | 0.48 s | - | 0.9 s | 4.4 GB / 6.3 GB |

  Remeasured on an idle machine after the review fixes, through `VoiceManager` (construction
  loads, quantizes and primes): 1.7B 8-bit construction 0.6 s, RTF 0.17-0.22 buffered and
  0.22-0.24 streamed, first audio 0.10-0.14 s; 0.6B 8-bit construction 0.4 s, RTF 0.19-0.23,
  first audio 0.09-0.12 s. `mx.get_peak_memory()` after load: 7.9 GB (1.7B) and 3.6 GB (0.6B),
  a load-time transient from reading bf16 weights before quantizing the talker, which the
  "Peak RSS" column above does not capture. GPU use by other processes does not show in the CPU
  load average and slowed the 1.7B about tenfold in one contended run.

  For comparison, the PyTorch path on the same machine the same night: 1.7B RTF 1.7-1.9 (load
  59 s), and in the quieter earlier measurement RTF 0.9-1.2 with first streamed audio at 9.1 s.
- Priming at load: the first utterance after a primed load starts in 0.10 s (1.7B) versus 0.19 s
  unprimed, and completes 0.4 s sooner; the very first process on a machine pays 2-3 s of Metal
  compilation in the prime (0.1 s once the OS shader cache holds the kernels).
- Parity with the PyTorch model (0.6B CustomVoice, MLX bf16 vs torch fp32): identical prompt
  layout (31 embeddings, max |diff| 0.007 = bf16 rounding), identical prefill argmax and top-5
  (cosine 0.99996), codec decoder SNR 49 dB, speaker embedding identical (cosine 1.0), reference
  codes 98.5% identical (first three codebooks 100%). Whisper (`large-v3-turbo`) WER on the
  63-word text, 1.7B, 3 samples each: PyTorch 0.17 mean, MLX 0.14 mean (floor ~0.11 from Whisper's
  own number/name normalisations). The tokenizer is token-identical to Transformers on every probe.
- No style drift: a multi-sentence reply is ONE generation; chunk sizes are whole codec frames
  (6 x 1920 - 120 samples, then 12 x 1920, then a 120-sample tail) with fades only at the edges.
- The 8-bit default stands: bf16 is memory-bound (RTF 0.92 for the 1.7B). No converted-weight cache:
  loading plus quantization is under 0.4 s with warm file pages and a few seconds cold.

### Files touched

- `abstractvoice/qwen3_tts/mlx/{__init__,_cache,config,talker,codec,speaker_encoder,tokenizer,weights,model}.py` (new)
- `abstractvoice/qwen3_tts/backend.py` (new), `abstractvoice/qwen3_tts/runtime.py`
- `abstractvoice/adapters/tts_qwen3_tts.py`, `abstractvoice/vm/tts_mixin.py`,
  `abstractvoice/cloning/engine_qwen3_tts.py`, `abstractvoice/engine_runtime.py`
- `pyproject.toml`, `third_party_licenses/mlx_audio_notice.txt`, `ACKNOWLEDGMENTS.md`
- `tests/test_qwen3_tts_backend_selection.py`, `tests/test_qwen3_tts_mlx_tokenizer.py`,
  `tests/test_qwen3_tts_mlx_e2e_optional.py` (new); `tests/test_qwen3_tts_predictor.py`,
  `tests/test_transformers_floor.py`, `tests/test_engine_runtime.py` (pinned to the PyTorch path /
  hide `mlx`)
- `README.md`, `docs/api.md`, `docs/installation.md`, `docs/dependencies.md`,
  `docs/troubleshooting.md`, `docs/architecture.md`, `docs/benchmark.md`, `CHANGELOG.md`,
  `llms.txt`, `llms-full.txt`, `docs/adr/0008_qwen3_tts_backend_selection.md`, `docs/adr/README.md`

### Validation

- `python -m pytest -q` -> 688 passed, 4 skipped, 66 deselected (default markers).
- `python -m pytest -q tests/test_qwen3_tts_backend_selection.py tests/test_qwen3_tts_mlx_tokenizer.py` -> 16 passed.
- `ABSTRACTVOICE_RUN_QWEN3_TTS_MLX_TESTS=1 python -m pytest -q -m model_download tests/test_qwen3_tts_mlx_e2e_optional.py` -> 5 passed (CustomVoice load/prime/stream/WER, VoiceDesign, Base clone from a 3 s reference + x-vector mode).
- PyTorch-path tests (`tests/test_qwen3_tts_predictor.py`, `tests/test_transformers_floor.py`) pass with the backend pinned to `torch`.

### Follow-ups

- LLM-delta streaming (`open_tts_text_stream`) still synthesizes per text segment as text arrives;
  the model's streaming text mode could feed later segments into the same generation.
- Fused QKV / gate-up projections are adopted. Compiling the code-predictor loop as one graph
  measured a further gain in the optimisation study (together with the fused projections, 15-25%
  per frame); not adopted yet.
- A shorter first chunk (2, then 4, then 8 frames) measured first audio at 0.07-0.10 s (1.7B) in the
  optimisation study without hurting throughput; the shipped cadence is 6 then 12.
- The codec encoder's deeper residual codebooks differ from PyTorch on ~1-3% of frames (fp32
  argmin ties); audibly irrelevant, noted for parity work.
