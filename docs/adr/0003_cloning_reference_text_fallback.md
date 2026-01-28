## ADR 0003: Voice cloning reference text (automatic fallback strategy)

**Date**: 2026-01-23  
**Status**: Accepted (amended 2026-01-28)  

---

## Context

Our cloning engines condition on a **reference transcript** + the generation text:

- `f5_tts`: **`ref_text + gen_text`**
- `chroma`: `prompt_text` (reference transcript) + `prompt_audio` (reference audio)

This means `reference_text` is not “metadata”: it directly influences generation.

If `ref_text` is wrong (e.g. noisy ASR), the model can:

- “bleed” wrong words into the output (hallucinated repetitions)
- degrade speaker similarity and prosody

We want a high-quality, automatic experience even when the user does **not** provide `reference_text`.

---

## Decision

### 1) Store `reference_text` per cloned voice and treat it as a first-class quality control input

- Each cloned voice in `VoiceCloneStore` stores `reference_text`.
- The system can update it once and reuse it for all future utterances.

### 2) `reference_text` is optional user input; if present, it overrides auto-STT

- Users may provide `reference_text` at clone time, or set it later.
- If user-provided text exists, we **do not** run STT auto-fallback.

### 3) Default fallback when `reference_text` is missing: one-time auto-generation, then persist

When `reference_text` is missing:

- Generate it **once** (not per utterance) via STT.
- Apply conservative normalization (punctuation, whitespace).
- Persist it into the voice store (with metadata marking it as auto-generated).

This avoids repeating the “poisoned prompt” problem across utterances.

Implementation note:

- We do this lazily at **first speak** (not at clone-time) so cloning remains offline-first and does not force
  STT downloads during `/clone ...` flows.
- This fallback is **engine-agnostic** and must behave the same for `f5_tts` and `chroma`.

### 4) STT strategy for auto-generated `reference_text`: 3-pass consensus

To reduce occasional ASR instability, we run STT **3 times** and select a consensus transcript:

- Normalize each candidate (whitespace + sentence-ending punctuation).
- If a majority exists (2/3 match after normalization), choose it.
- Otherwise, choose the “closest” candidate (minimum total edit distance to the others).

If STT produces no usable text, we fail with a clear error and require the user to set `reference_text` manually.

### 5) Optional “quality mode” (future): closed-loop selection of ref_text candidates

If we need better robustness than single-pass ASR:

- Generate several candidate transcripts (different STT models/decoding/normalizations).
- Synthesize a short probe phrase for each candidate.
- Select the candidate that maximizes speaker similarity vs reference audio (speaker embeddings).

This optimizes the true objective (speaker similarity) rather than ASR accuracy alone.

---

## Consequences

- We accept that fully automatic fallback is probabilistic, but we constrain the blast radius
  by persisting a best-effort transcript once and enabling easy correction.
- We prefer deterministic behavior (one-time preprocessing + caching) over silent repeated
  re-transcription.

---

## Related

- `docs/adr/0002_barge_in_interruption.md`
- Backlog: `docs/backlog/completed/020_cloning_reference_text_autofallback.md`
