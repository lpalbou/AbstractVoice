# Task 059: True preload/unload for local STT + TTS models (AbstractCore residency compliance)

**Date**: 2026-05-21  
**Status**: Completed  
**Priority**: P1  
**Promoted from**: `docs/backlog/planned/0059_local_model_preload_and_residency.md`  
**Completed**: 2026-05-21  

---

## Main goals

- Make local **TTS** and local **STT** engines support a **real in-process preload** concept (weights loaded, ready to run) and a best-effort **unload** that releases memory.
- Make AbstractVoice’s AbstractCore capability plugin residency surface usable for local audio:
  - `load_resident_model(...)`
  - `list_resident_models(...)`
  - `unload_resident_model(...)`

## Secondary goals

- Keep discovery import-light (do not load models during `available_providers` / `list_models`).
- Preserve existing `VoiceManager` public behavior for normal speak/transcribe flows.
- Provide a repeatable benchmark showing the benefit of preload vs cold-start.

---

## Context / problem

AbstractVoice already implements AbstractCore discovery for voice + audio backends (providers/models), and cloned-TTS residency warmup exists. However, the “resident model” concept was not previously implementable for **base local TTS** and **local STT** engines in a robust way:

- there was no first-class `VoiceManager.preload_*` / `unload_*` surface for local engines;
- the capability plugin residency methods could not reliably report resident state for local TTS/STT.

This blocks server/process startup warmup patterns and makes it hard for AbstractCore (or a UI) to separate:

- cold start (load + first generation/transcription)
- preloaded steady-state generation/transcription time

---

## Constraints

- Residency preload/unload is **local engines/models only** (remote OpenAI/OpenAI-compatible providers remain “configured”, not “resident”).
- Discovery methods must remain import-light and **must not load weights**.
- Unload must be best-effort and must not permanently break the `VoiceManager` instance (a subsequent request should be able to re-load).
- Avoid coupling to a single adapter; implement through `VoiceManager` orchestration plus best-effort adapter hooks.

---

## Research, options, and references

### Option A — Plugin-only “resident” bookkeeping (no real preload)

- Track resident state in the AbstractCore plugin cache without actually loading models.
- Pros: minimal changes, no adapter changes.
- Cons: not “true preload”; cannot improve latency; `unload` is meaningless.

### Option B — VoiceManager-level preload/unload + plugin delegation (chosen)

- Implement `preload_tts_engine`, `unload_tts_engine`, `preload_stt_engine`, `unload_stt_engine` on `VoiceManager`.
- Add `list_resident_components()` reporting for clone engines + base TTS + STT (best-effort).
- Have the AbstractCore capability plugin call these methods for local engines only.
- Pros: real model residency and measurable latency improvement; single orchestration point; plugin stays thin.
- Cons: requires careful best-effort unload across several runtimes.

Key local engine load-state signals:

- Piper: adapter `_voice` object indicates loaded voice weights/session.
- Supertonic: runtime `_loaded` (and cleared session objects on unload).
- OmniVoice / AudioDiT: runtime `_model` indicates loaded.
- Faster-Whisper / Transformers ASR: adapter maintains `_model` / `_pipeline` and can be dropped on unload.

---

## Decision

**Chosen approach**: implement VoiceManager-driven local TTS/STT preload/unload and report resident components; wire AbstractCore plugin residency methods to these functions for local engines only.

**Why**:
- keeps discovery/model catalogs import-light;
- provides a real performance win (steady-state generation/transcribe time);
- avoids having AbstractCore-specific logic leak into each adapter/runtime.

---

## Dependencies

- Backlog:
  - Completed: `docs/backlog/completed/0057_abstractcore_plugin_audio_discovery_surface.md`
  - Completed: `docs/backlog/completed/0056_normalize_abstractcore_capability_residency_truth.md`

---

## Implementation plan

- Add `VoiceManager` methods:
  - `preload_tts_engine(...)` and `unload_tts_engine()`
  - `preload_stt_engine(...)` and `unload_stt_engine()`
  - `list_resident_components()`
- Update AbstractCore capability plugin:
  - Voice backend (`abstractvoice:default`): support local TTS residency (in addition to cloning).
  - Audio backend (`abstractvoice:stt`): support local STT residency.
- Add a small benchmark script to measure cold vs preloaded behavior (runs=3).
- Add/extend unit tests for the plugin residency contract (no heavy model loads).

---

## Success criteria

- `core.voice.load_resident_model({task:\"tts\", provider:\"piper\"|\"omnivoice\"|...})` returns `loaded=true` only when the local engine is actually loaded.
- `core.audio.load_resident_model({task:\"stt\", provider:\"faster-whisper\"|\"transformers-asr\"})` returns `loaded=true` for local STT engines.
- `list_resident_models(...)` surfaces resident local TTS/STT engines without loading anything new.
- `unload_resident_model(...)` makes resident engines non-resident and frees memory best-effort.
- Benchmarks show meaningfully lower steady-state generation time (`hot_synth`) vs cold start.

---

## Test plan

- `python -m pytest -q -o faulthandler_timeout=30`
- `python examples/bench_preload_local_models.py --runs 3`

---

## Report

### Summary

- Added a real “preloaded” concept for **local base TTS** and **local STT**:
  - `VoiceManager.preload_tts_engine(...)` / `VoiceManager.unload_tts_engine()`
  - `VoiceManager.preload_stt_engine(...)` / `VoiceManager.unload_stt_engine()`
  - `VoiceManager.list_resident_components()` now reports clone engines + base TTS + STT (best-effort, local-only).
- Wired AbstractCore capability plugin residency methods to these VoiceManager methods:
  - voice backend (`abstractvoice:default`) supports local TTS residency (in addition to clone residency),
  - audio backend (`abstractvoice:stt`) supports local STT residency.
- Added an end-to-end benchmark script comparing cold start vs preloaded steady-state latency and captured averaged results (3 runs).
- Fixed Piper Spanish model mapping to a valid upstream voice (`es_ES-davefx-medium`) to keep the curated Piper set downloadable/offline-cacheable.

### Benchmarks (local machine, 3 runs)

Command:

`python examples/bench_preload_local_models.py --runs 3`

TTS (avg seconds):

| Provider | Model | Cold avg | Preload avg | Hot avg | Speedup (cold/hot) |
| --- | --- | ---: | ---: | ---: | ---: |
| piper | en_US-amy-medium | 0.394 | 0.348 | 0.036 | 11.02x |
| piper | fr_FR-siwis-medium | 0.348 | 0.349 | 0.036 | 9.52x |
| piper | de_DE-thorsten-medium | 0.346 | 0.346 | 0.034 | 10.06x |
| piper | es_ES-davefx-medium | 0.348 | 0.351 | 0.034 | 10.19x |
| piper | ru_RU-dmitri-medium | 0.345 | 0.348 | 0.036 | 9.69x |
| piper | zh_CN-huayan-medium | 0.345 | 0.360 | 0.039 | 8.77x |
| supertonic | supertonic-3 | 0.582 | 0.572 | 0.395 | 1.48x |
| omnivoice | default | 0.884 | 0.806 | 0.249 | 3.55x |
| audiodit | default | 3.301 | 3.451 | 0.600 | 5.50x |

STT (avg seconds):

| Provider | Model | Cold avg | Preload avg | Hot avg | Speedup (cold/hot) |
| --- | --- | ---: | ---: | ---: | ---: |
| faster-whisper | base | 0.730 | 0.090 | 0.570 | 1.28x |
| transformers-asr | CohereLabs/cohere-transcribe-03-2026 | 34.830 | 26.204 | 0.793 | 43.91x |

Notes:
- “Cold avg” includes load + first run. “Hot avg” is steady-state once preloaded.
- Results are hardware- and cache-dependent; relative direction is the intended signal.

### Validation

- `python -m pytest -q -o faulthandler_timeout=30` (206 passed, 3 skipped, 32 deselected)
- `python examples/bench_preload_local_models.py --runs 3` (results above)
