## ADR 0004: Streaming + cancellation for cloned TTS (REPL responsiveness)

**Date**: 2026-01-23  
**Status**: Accepted  

---

## Context

Cloned TTS is computationally heavier than Piper. Even after eliminating per-utterance
model reload, long answers can still take multiple seconds.

For UX, the key metric is often **time-to-first-audio** and **interruptibility**:

- start speaking quickly (even if the full answer takes longer)
- allow immediate interruption on new user input (barge-in by typing / speaking)

---

## Decision

### 1) Stream audio chunks into the existing non-blocking player

- Generate audio in smaller text batches (sentence/word-bounded chunking).
- As soon as a batch completes, enqueue it to `NonBlockingAudioPlayer`.
- This provides “speak early” behavior without requiring the model to be truly streaming.

### 2) Add cancellation/preemption

- Track a per-speak “cancel token” for cloned synthesis.
- `stop_speaking()` and new user input set the token, stopping:
  - queued playback immediately
  - further batch enqueueing as soon as the generator checks the token

Note: model sampling is not always preemptible mid-step; we implement best-effort
cancellation between batches/chunks.

---

## Consequences

- Better perceived latency: users hear speech sooner.
- Cleaner REPL behavior: new input can interrupt current speech.
- Implementation stays within existing architecture (reuse audio player and callbacks).

---

## Related

- `docs/adr/0002_barge_in_interruption.md` (Phase 2 AEC enables true barge-in)
- Backlog: `docs/backlog/planned/021_streaming_and_cancellation_for_cloned_tts.md`

