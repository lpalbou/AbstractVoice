## Task 031: Improve Chroma cloning quality (speaker similarity + stability)

**Date**: 2026-01-28  
**Status**: Planned  
**Priority**: P1  

---

## Main goals

- Improve **speaker similarity** of the `chroma` cloning backend so it is competitive with `f5_tts` on common samples.
- Make Chroma cloning behavior more consistent with upstream expectations (prompt length, preprocessing, decoding knobs).

## Secondary goals

- Keep the cloning API engine-agnostic (ADR 0003): `reference_text` remains optional; auto-STT fallback works for all engines.
- Add a repeatable evaluation harness (even if “lightweight”) to avoid subjective-only tuning.

---

## Context / problem

Chroma inference now works end-to-end in AbstractVoice, but in practice speaker similarity can be noticeably worse than
`f5_tts`, especially when users provide short prompts (e.g., ~5s).

Upstream Chroma examples appear to rely on **longer prompt audio** (≈ 13–26s), and their processor/generation logic has
specific expectations:

- Prompt audio is loaded with torchaudio and resampled to 24kHz; upstream does not aggressively normalize/trim.
- Voice cloning relies on `prompt_audio` + `prompt_text` alignment. If the transcript does not match the audio, cloning quality drops.
- Upstream “audio generation” uses **top-k sampling + temperature** (their custom generation code ignores top-p).

Because AbstractVoice aims for “clone from whatever sample you have”, we need a strategy that works reasonably well even
when prompt audio is short or messy — without breaking offline-first constraints.

---

## Constraints

- Keep `VoiceManager` + `VoiceCloner` API stable.
- Do not add heavy mandatory deps to the base install.
- Maintain offline-first REPL behavior (no surprise downloads).
- Respect ADR 0003: `reference_text` is optional; auto-STT fallback must remain engine-agnostic.

---

## Research, options, and references

This section is self-contained so implementation is straightforward.

### Observations from upstream Chroma

- Upstream repo includes example prompt audios that are **much longer than 6–10s**:
  - `/Users/albou/projects/gh/FlashLabs-Chroma/example/prompt_audio/*.wav` are ~13–26s (48kHz).
- Upstream processor:
  - Loads prompt audio with torchaudio, resamples to 24k, no additional normalization:
    - `/Users/albou/projects/gh/FlashLabs-Chroma/chroma/processing_chroma.py`
- Upstream generation:
  - Custom `sample_topk(logits, top_k, temperature)` is used; `top_p` is not applied:
    - `/Users/albou/projects/gh/FlashLabs-Chroma/chroma/generation_chroma.py`
- Model expects prompt audio cutoffs in samples and builds an attention mask over encoded audio frames:
  - `/Users/albou/projects/gh/FlashLabs-Chroma/chroma/modeling_chroma.py`

References:
- Model repo: `https://huggingface.co/FlashLabs/Chroma-4B` (may be gated)
- Upstream code: `https://github.com/FlashLabs-AI-Corp/FlashLabs-Chroma`
- Paper: `https://arxiv.org/abs/2601.11141`

### Option A: Make Chroma prompt audio “better” by allowing multi-file prompts (merge/concat)

Problem addressed:
- Chroma currently expects a single prompt audio file; users often have multiple short clips.

Approach:
- Allow cloning with directories / multiple reference files for `engine="chroma"` (like `f5_tts`).
- At inference time, merge references into a single prompt WAV:
  - resample to 24kHz
  - mono downmix
  - optional mild silence trimming (VAD/energy) at boundaries only
  - cap total length to ~30s for stability
- Cache the merged prompt file per voice_id (and invalidate on reference file change).

Trade-offs:
- Improves similarity by providing more conditioning signal.
- Adds a small amount of preprocessing complexity, but no new deps (can reuse torchaudio when installed; otherwise fallback).

### Option B: Improve `reference_text` quality specifically for Chroma *without violating ADR 0003*

Problem addressed:
- Chroma is sensitive to `prompt_text` alignment; STT errors reduce similarity.

Approach (engine-agnostic implementation, but tuned with Chroma in mind):
- Ensure auto-STT uses the highest-quality available decoding path (file-based faster-whisper w/ VAD + beam search).
- Increase the default max transcription window for reference prompts (e.g., up to 30s).
- Add a “quality mode” flag later:
  - allow a larger STT model (`medium` / `large-v3`) for reference_text generation only
  - keep default small to avoid surprises

Trade-offs:
- Better transcripts improve Chroma a lot, but can still fail on noisy prompts.
- Larger STT models add time and disk; must remain opt-in for offline-first.

### Option C: Add a reproducible “speaker similarity” evaluation harness

Problem addressed:
- Hard to tune Chroma by feel; we need objective metrics.

Approach:
- Add an optional script that:
  - synthesizes a fixed set of probe texts for a voice (F5 + Chroma)
  - computes speaker embeddings similarity vs reference audio
    - Use an optional dependency group for speaker embeddings (keep it out of core).
- Store results as a simple JSON report under `docs/reports/`.

Trade-offs:
- Requires an additional model/dep for embeddings and may be GPU-heavy.
- But it provides an engineering feedback loop for quality tuning.

---

## Decision

**Chosen approach**: Start with **Option A** + **Option B** (practical quality wins without new heavyweight deps). Add Option C only if we plan to actively tune Chroma.

**Why**:
- Upstream evidence strongly suggests prompt length matters (their examples are long).
- Alignment between `prompt_audio` and `prompt_text` is critical for cloning quality.
- These changes are compatible with ADR 0003 and keep the REPL offline-first.

---

## Dependencies

- **ADRs**:
  - `docs/adr/0003_cloning_reference_text_fallback.md`
- **Backlog tasks**:
  - Planned: `docs/backlog/planned/030_chroma_4b_optional_s2s_and_cloning.md`
  - This task: `docs/backlog/planned/031_chroma_cloning_quality_improvements.md`

---

## Implementation plan

- Allow `VoiceCloner.clone_voice(..., engine="chroma")` to accept directories / multiple refs.
- Add a Chroma-specific “prompt merge” helper (engine-side) that:
  - concatenates multiple refs into one cached prompt wav (<= 30s @ 24kHz PCM16 mono).
- Improve the auto `reference_text` pipeline (engine-agnostic):
  - prefer higher-quality STT decode path and a longer window (<= 30s).
- Expose a Chroma quality preset:
  - tune `temperature` and `top_k` defaults; consider greedy mode for stability.
- Update docs (REPL + cloning docs) with guidance:
  - Chroma prefers longer, clean prompts; short prompts may degrade similarity.

---

## Success criteria

- On internal samples (e.g., HAL), Chroma produces noticeably better similarity when using a folder of multiple clips vs a single 5–6s clip.
- When auto `reference_text` is used, Chroma similarity is not significantly worse than when the user manually sets the transcript (for clean prompts).

---

## Test plan

- `pytest -q`
- Manual (GPU/MPS):
  - Clone Chroma from:
    - a single short file (~5–6s)
    - a directory containing multiple clips merged to ~20–30s
  - Compare output similarity subjectively + ensure no crashes and stable output duration.

---

## Report (fill only when completed)

### Summary

<what changed and why>

### Validation

- Tests: <result>
