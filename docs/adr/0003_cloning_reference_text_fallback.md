## ADR 0003: Voice cloning reference text (automatic fallback strategy)

**Date**: 2026-01-23  
**Status**: Accepted  

---

## Context

Our cloning engine (F5-style prompted TTS) conditions on **`ref_text + gen_text`**.
This means `ref_text` is not “metadata”: it directly influences generation.

If `ref_text` is wrong (e.g. noisy ASR), the model can:

- “bleed” wrong words into the output (hallucinated repetitions)
- degrade speaker similarity and prosody

We want a high-quality, automatic experience even when the user does not provide `ref_text`.

---

## Decision

### 1) Store `reference_text` per cloned voice and treat it as a first-class quality control input

- Each cloned voice in `VoiceCloneStore` stores `reference_text`.
- The system can update it once and reuse it for all future utterances.

### 2) Default fallback when `reference_text` is missing: one-time auto-generation, then persist

When `reference_text` is missing:

- Generate it **once** (not per utterance) via STT.
- Apply conservative normalization (punctuation, whitespace).
- Persist it into the voice store (with metadata marking it as auto-generated).

This avoids repeating the “poisoned prompt” problem across utterances.

### 3) Optional “quality mode” (future): closed-loop selection of ref_text candidates

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
- Backlog: `docs/backlog/planned/020_cloning_reference_text_autofallback.md`

