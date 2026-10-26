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

MEASURED_SUMMARY_PLACEHOLDER

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

MEASURED_VALIDATION_PLACEHOLDER

### Follow-ups

- LLM-delta streaming (`open_tts_text_stream`) still synthesizes per text segment as text arrives;
  the model's streaming text mode could feed later segments into the same generation.
- Fused QKV / gate-up projections and a compiled code-predictor loop measured ~20% faster per
  frame in the optimisation study; worth adopting if a tighter real-time factor is needed.
- The codec encoder's deeper residual codebooks differ from PyTorch on ~1-3% of frames (fp32
  argmin ties); audibly irrelevant, noted for parity work.
