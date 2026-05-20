# ADR 0003: Cloning reference text is first-class and auto-filled once

Status: Accepted. Amended 2026-01-28.

## Context

Several cloning engines depend on both reference audio and a reference
transcript. That transcript is not decorative metadata. It directly affects
speaker similarity, intelligibility, and prompt stability.

If the transcript is wrong, the system can bleed incorrect words into the
output, degrade prosody, or repeatedly pay the cost of re-transcribing the same
reference. Users should be allowed to provide the transcript themselves, but the
default experience still needs a robust automatic fallback.

## Decision

- `reference_text` is a first-class property of a stored cloned voice.
- User-provided `reference_text` wins over automatic STT and is persisted for
  reuse.
- When `reference_text` is missing, AbstractVoice generates it once, persists
  it, and reuses it for later synthesis instead of retranscribing per
  utterance.
- The automatic fallback is engine-agnostic at the manager/store layer. Clone
  engines should consume the stored transcript rather than inventing their own
  private fallback rules.
- Automatic fallback uses conservative normalization and a 3-pass STT consensus
  strategy before persistence.
- If no usable transcript can be produced, the system fails clearly and asks for
  a better sample or manual text instead of silently proceeding with degraded
  prompts.

## Consequences

### Positive

- Clone quality becomes more stable across utterances.
- Users can fix transcript quality once and get the benefit on all future
  synthesis.
- The store captures provenance such as manual versus ASR-generated text.

### Negative

- The first synthesis for a newly created voice may pay a one-time STT cost.
- The automatic fallback remains probabilistic and can still be wrong on noisy
  references.

### Neutral

- The fallback is intentionally manager-owned rather than engine-owned, so clone
  engines stay simpler and more consistent.

## Enforcement

- Clone engines that require transcript conditioning must consume
  store-managed `reference_text` passed through `VoiceCloner`.
- Do not re-transcribe reference audio on every utterance.
- Preserve `reference_text_source` provenance in the store when the text is set
  manually or via ASR.
- Any change to the automatic fallback strategy must keep the engine-agnostic
  contract or explicitly supersede this ADR.

## Validation

- `tests/test_cloning_reference_text_autofallback.py`
- `tests/test_reference_text_autofallback_offline_cached_model.py`
- `tests/test_voice_cloning_store.py`

## Backlog links

- `docs/backlog/completed/020_cloning_reference_text_autofallback.md`
- `docs/backlog/planned/031_chroma_cloning_quality_improvements.md`

## Related

- [0004_streaming_and_cancellation_for_cloned_tts.md](./0004_streaming_and_cancellation_for_cloned_tts.md)
- `abstractvoice/cloning/manager.py`
- `abstractvoice/cloning/store.py`
