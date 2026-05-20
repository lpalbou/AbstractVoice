# ADR Index

Accepted ADRs in this folder are live engineering policy for AbstractVoice.
Update the relevant ADR, this index, and any linked backlog items together when
the policy changes.

## Catalog

- [ADR 0001](./0001-local_assistant_out_of_box.md) — Accepted. Amended 2026-05-08.
  Defines the out-of-box contract for explicit local assistant installs while
  keeping the base package remote-first.
- [ADR 0002](./0002_barge_in_interruption.md) — Accepted. Amended 2026-02-04.
  Defines voice-mode and interruption semantics, including stop-phrase behavior
  and optional AEC.
- [ADR 0003](./0003_cloning_reference_text_fallback.md) — Accepted. Amended 2026-01-28.
  Makes cloning `reference_text` first-class and requires one-time
  auto-generation plus persistence when missing.
- [ADR 0004](./0004_streaming_and_cancellation_for_cloned_tts.md) — Accepted.
  Requires cloned TTS to favor early playback and per-utterance cancellation.
- [ADR 0005](./0005_torch_device_and_dtype_policy.md) — Accepted.
  Makes shared compute helpers the default source of truth for torch device and
  dtype selection.
- [ADR 0006](./0006_voice_overlays_are_package_owned_not_generic_adapters.md) — Accepted.
  Keeps provider/model/profile/voice/quality/residency semantics package-owned
  rather than flattening them into a generic adapter metaphor.
- [ADR 0007](./0007_directed_speech_requests_and_planning_are_package_owned.md) — Accepted.
  Defines the package-owned directed speech contract, explicit capability
  degradation, and the planning boundary that stays inside AbstractVoice.

## Working rules

- Keep `Context` and `Decision` first.
- Include `Consequences`, `Enforcement`, and `Validation` for accepted ADRs.
- Use backlog items for execution history and open work; use ADRs for durable
  rules that constrain future work.
