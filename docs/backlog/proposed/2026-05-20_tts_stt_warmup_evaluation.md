# TTS/STT Warmup Evaluation

## Summary

Do **not** add generic Runtime/Gateway/Flow model-residency warmup for base TTS/STT yet.

The current local `abstractvoice` paths are already light enough that the gain is marginal for normal usage:

- Piper: small init cost, very small first-request delta
- Supertonic: modest first-request delta, still small in absolute time
- STT base path: no evidence yet that a generic cross-framework warmup abstraction would pay for its complexity

Voice cloning is different and already has a heavier explicit preload surface.

## Recommendation

### 1. No generic TTS/STT residency control for base voice/transcription

Do not mirror image-model warmup across the whole stack for:

- Piper
- Supertonic
- ordinary STT adapters

The likely latency win is too small to justify extra control-plane and UX complexity.

### 2. Keep warmup where it is materially useful

Heavy paths such as cloning engines may continue to expose their own explicit preload behavior.

### 3. Revisit only with measured evidence

Re-open this only if one of these becomes true:

- a first-request penalty exceeds roughly 10% in real user flows
- a new local TTS/STT backend has multi-second cold-start cost
- cloning-heavy voice workflows become a primary everyday path in Flow/Gateway

## Acceptance Criteria

- No new generic TTS/STT residency endpoints or workflow controls are added now.
- Documentation points workflow authors toward cloning-specific preload where relevant.
- Future reconsideration requires measured cold-vs-warm evidence, not symmetry with image warmup.
