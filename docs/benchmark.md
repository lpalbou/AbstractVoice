# Benchmark (TTS-only, no voice cloning)

This document provides **empirical TTS timing results** for AbstractVoice’s available TTS engines, plus practical notes about **voice stability** (reusing the “same voice” across turns) and **language support** (English/French).

Benchmarks are **hardware- and configuration-dependent**. Treat the numbers as **relative comparisons** and re-run on your target machine for decisions.

## Quick takeaways (what to use)

- **If you need the fastest, simplest English/French TTS**: use **Piper** (default).
- **If you need omnilingual TTS and a “designed voice” you can keep stable with a seed**: use **OmniVoice**.
- **If you’re focused on EN/ZH and longer prompts (and accept heavier deps)**: **AudioDiT** can scale well, but French is not a supported target in this integration.

This benchmark measures **TTS only**. Voice cloning quality is **not** benchmarked here.

## What we measure

- **WAV generation time (seconds)**: end-to-end wall time of `VoiceManager.speak_to_file(..., format="wav")`.
- **Speech duration (seconds)**: duration of the generated WAV file.
- **RTF (“real-time factor”)**: \(RTF = \frac{\text{generation time}}{\text{speech duration}}\).
  - `RTF < 1.0` means faster than real time (good).
  - `RTF = 0.1` means “~10× faster than real time”.
- **Multiple experiments**: we run the same benchmark several times to capture run-to-run variance (thermal/scheduler/accelerator effects).

Where it’s defined:
- These values are computed in `examples/bench_tts.py` and aggregated across runs by `examples/bench_tts_suite.py`.

## Engines covered

TTS engines in this repo (see `abstractvoice/adapters/tts_registry.py`):

- **Piper** (`tts_engine="piper"`) — default, lightweight, deterministic voice model selection
- **AudioDiT** (`tts_engine="audiodit"`) — torch/transformers (LongCat-AudioDiT-1B by default)
- **OmniVoice** (`tts_engine="omnivoice"`) — torch/transformers (k2-fsa/OmniVoice)

AudioDiT note (model size):
- This benchmark uses the default checkpoint: `meituan-longcat/LongCat-AudioDiT-1B`.
- Upstream also provides a larger checkpoint: `meituan-longcat/LongCat-AudioDiT-3.5B` (not benchmarked here; expect higher memory/compute).

## Voice stability (“same voice across a discussion”)

- **Piper**: stable by design.
  - Reuse the same `language` and Piper voice mapping (AbstractVoice uses one default voice per language in the shipped mapping).
- **AudioDiT**: best-effort stability.
  - Uses a **seed** (`AudioDiTSettings.seed`) plus a **session prompt** mechanism in `AudioDiTTTSAdapter` (enabled by default) to keep a consistent speaker across turns.
- **OmniVoice**: best-effort stability.
  - Use a fixed **`instruct`** (voice design attributes) and **`seed`** (plus keep temperatures the same).
  - Note: torch accelerators (MPS/CUDA) can still introduce small nondeterminism. If you need stricter determinism, prefer CPU float32.

## English vs French support (important)

- **Piper**: English + French are supported in the shipped catalog (`en`, `fr`).
- **OmniVoice**: upstream is designed for omnilingual TTS; `language="en"` / `language="fr"` are supported.
- **AudioDiT**: upstream focus is EN/ZH. The adapter **declares** supported languages `["en","zh"]`.
  - French synthesis may still “work” but is **not guaranteed** to be intelligible or high quality.

## How to reproduce

### Single experiment (one run with N repetitions per text length)

```bash
python examples/bench_tts.py \
  --out-dir untracked/bench_tts_rep5_s2s5 \
  --engines piper,audiodit,omnivoice \
  --languages en,fr \
  --variants s2,s5 \
  --repetitions 5
```

Outputs:
- `results.json`: per-run timings + WAV paths
- `summary.md` / `summary.json`: mean/std per (engine, language, variant)

### Multi-experiment suite (recommended)

Runs the full benchmark **in separate Python processes** multiple times, then aggregates:

```bash
python examples/bench_tts_suite.py \
  --out-dir untracked/bench_tts_suite_reps5_2026-04-06 \
  --experiments 5 \
  --sleep-s 2 \
  --engines piper,audiodit,omnivoice \
  --languages en,fr \
  --variants s2,s5 \
  --repetitions 5
```

Outputs:
- `suite_summary.md` / `suite_summary.json`: aggregated mean/std + between-experiment variance
- `exp_*/`: each experiment run’s raw WAVs + per-run results

## Example results (macOS / Apple Silicon, MPS fp16) — 5 experiments × 5 reps

These results were collected with:
- torch device: `mps`
- torch dtype: `float16`
- 5 experiments × 5 repetitions per condition → **25 WAV generations per condition**

This is intentionally “order of magnitude” oriented. If you need variance details (standard deviation / min / max), run the suite yourself and inspect the generated `suite_summary.md`.

| engine | lang | text | avg WAV gen time (s) | avg speech length (s) | RTF |
|---|---|---|---:|---:|---:|
| **piper** | en | 2 sentences | **0.080** | 4.237 | 0.019 |
| **piper** | fr | 2 sentences | **0.079** | 4.282 | 0.019 |
| **piper** | en | 5 sentences | **0.255** | 14.021 | 0.018 |
| **piper** | fr | 5 sentences | **0.285** | 15.639 | 0.018 |
| **omnivoice** | en | 2 sentences | 0.769 | 3.560 | 0.216 |
| **omnivoice** | fr | 2 sentences | 0.863 | 4.000 | 0.216 |
| **omnivoice** | en | 5 sentences | 2.199 | 12.240 | 0.180 |
| **omnivoice** | fr | 5 sentences | 3.097 | 16.520 | 0.187 |
| **audiodit** | en | 2 sentences | 0.710 | 7.680 | 0.092 |
| **audiodit** | fr | 2 sentences | 0.721 | 8.277 | 0.087 |
| **audiodit** | en | 5 sentences | 0.934 | 17.493 | 0.053 |
| **audiodit** | fr | 5 sentences | 1.205 | 22.528 | 0.053 |

Interpretation:
- **Piper** is fastest by orders of magnitude (sub-second generation for these prompts).
- Torch engines are slower on MPS for short prompts, but scale differently with longer prompts:
  - AudioDiT shows lower RTF for longer text in this run.
  - OmniVoice is competitive and has strong multilingual coverage.

## Subjective note (voice cloning)

This benchmark does **not** measure voice cloning.

Maintainer note (subjective): for **voice cloning**, OmniVoice tends to produce **impressively high quality** in practice and is often the best choice when cloning is the priority.

