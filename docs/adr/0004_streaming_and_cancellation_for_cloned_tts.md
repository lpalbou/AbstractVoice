# ADR 0004: Cloned TTS favors early playback and per-utterance cancellation

Status: Accepted.

## Context

Cloned TTS is materially heavier than the light local TTS paths. Users judge the
experience mainly by:

- how quickly they hear the first audio;
- whether new user input can stop the current response.

Waiting for a full cloned utterance to finish before playback makes the REPL and
interactive flows feel frozen even when the total synthesis time is acceptable.

## Decision

- Cloned TTS should prefer progressive playback over full-utterance buffering
  when the engine can support chunked output.
- The package may implement chunking by text batching and progressive enqueueing;
  it does not need to promise true token-level or diffusion-step streaming.
- Cancellation is per utterance. `stop_speaking()` and superseding user input
  must stop playback immediately and prevent future chunk enqueueing on a
  best-effort basis.
- The cancellation boundary is chunk/batch level unless an engine can do better.
  The package must not pretend mid-step model cancellation is universal.

## Consequences

### Positive

- Lower time-to-first-audio for cloned voices.
- Better interactive responsiveness in the REPL and browser flows.
- The policy fits the existing audio-player architecture instead of requiring a
  completely different playback stack.

### Negative

- Chunk boundaries can affect prosody compared with one-shot full-utterance
  synthesis.
- Cancellation cannot always interrupt a model inside its current inference
  step.

### Neutral

- Base TTS and cloned TTS are allowed to have different delivery implementations
  as long as their user-facing behavior stays honest.

## Enforcement

- Do not regress the clone path back to full buffered playback by default
  without revisiting this ADR.
- Keep cancellation state per utterance rather than as one global mutable flag.
- New clone engines should either implement chunked output or clearly degrade to
  buffered mode without claiming equivalent responsiveness.

## Validation

- `tests/test_cloned_tts_cancellation.py`
- `tests/test_voice_cloner_engine_dispatch.py`

## Backlog links

- `docs/backlog/completed/021_streaming_and_cancellation_for_cloned_tts.md`

## Related

- [0002_barge_in_interruption.md](./0002_barge_in_interruption.md)
- `abstractvoice/vm/tts_mixin.py`
- `abstractvoice/tts/tts_engine.py`
