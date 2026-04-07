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
  --out-dir untracked/bench_tts_suite_reps5_2026-04-06_fixed \
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

## Benchmark texts (fixed prompts)

Yes — the benchmark uses **fixed texts** per `(language, variant)` across **all engines** (defined in `examples/bench_tts.py`).

Warmup always uses `s2` (so adapters that build caches on first use pay that cost outside the averages).

### English (`en`)

- **`s2`**:
  - `Hello. This is a benchmark sentence for AbstractVoice.`
- **`s5`**:
  - `Hello. This is a benchmark sentence for AbstractVoice. We are measuring how long speech synthesis takes. This extra sentence increases the length of the prompt. Finally, this is the fifth sentence.`

### French (`fr`)

- **`s2`**:
  - `Bonjour. Ceci est une phrase de benchmark pour AbstractVoice.`
- **`s6`**:
  - `Bonjour. Ceci est une phrase de benchmark pour AbstractVoice. Nous mesurons le temps nécessaire pour générer la parole. Cette phrase supplémentaire augmente la longueur du texte. Nous ajoutons encore une phrase pour arriver à cinq. Enfin, voici la sixième phrase.`

## Example results (macOS / Apple Silicon, MPS fp16) — 5 experiments × 5 reps

These results were collected with:
- torch device: `mps`
- torch dtype: `float16`
- 5 experiments × 5 repetitions per condition → **25 WAV generations per condition**

Important clarification:
- The **torch device/dtype** settings apply to **torch-based engines** only (AudioDiT, OmniVoice, and most cloning engines).
- **Piper** does **not** use torch; it runs via **ONNX Runtime**. On macOS this is typically CPU-backed (still very fast).

This is intentionally “order of magnitude” oriented. If you need variance details (standard deviation / min / max), run the suite yourself and inspect the generated `suite_summary.md`.

| engine | lang | text | avg WAV gen time (s) | avg speech length (s) | RTF |
|---|---|---|---:|---:|---:|
| **piper** | en | s2 | **0.077** | 4.232 | 0.018 |
| **piper** | fr | s2 | **0.078** | 4.250 | 0.018 |
| **piper** | en | s5 | **0.246** | 14.023 | 0.018 |
| **piper** | fr | s6 | **0.276** | 15.819 | 0.017 |
| **omnivoice** | en | s2 | 0.889 | 3.560 | 0.250 |
| **omnivoice** | fr | s2 | 0.941 | 4.000 | 0.235 |
| **omnivoice** | en | s5 | 2.490 | 12.240 | 0.203 |
| **omnivoice** | fr | s6 | 3.579 | 16.520 | 0.217 |
| **audiodit** | en | s2 | 0.676 | 3.840 | 0.176 |
| **audiodit** | fr | s2 | 0.715 | 4.267 | 0.168 |
| **audiodit** | en | s5 | 1.072 | 13.653 | 0.079 |
| **audiodit** | fr | s6 | 1.443 | 18.517 | 0.078 |

Interpretation:
- **Piper** is fastest by orders of magnitude (sub-second generation for these prompts).
- Torch engines are slower on MPS for short prompts, but scale differently with longer prompts:
  - AudioDiT shows lower RTF for longer text in this run.
  - OmniVoice is competitive and has strong multilingual coverage.

## Subjective note (voice cloning)

This benchmark does **not** measure voice cloning.

Maintainer note (subjective): for **voice cloning**, OmniVoice tends to produce **impressively high quality** in practice and is often the best choice when cloning is the priority.

