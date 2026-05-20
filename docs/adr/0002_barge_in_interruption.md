# ADR 0002: Barge-in and voice-mode interruption semantics

Status: Accepted. Amended 2026-02-04.

## Context

Interactive voice systems feel broken when the assistant keeps talking after the
user has clearly tried to interrupt it. The hard part is that speaker playback
leaks into the microphone. Without echo cancellation, a naive "interrupt on any
speech" policy causes self-interruption and unstable turn-taking.

AbstractVoice therefore needs an interruption policy that is honest about the
difference between:

- strict turn-taking;
- stop-phrase interruption while the assistant is speaking;
- true speech-triggered barge-in.

## Decision

- Voice interruption is an explicit mode contract, not an implementation detail.
- The supported voice modes are:
  - `wait`: pause mic processing during TTS playback;
  - `stop`: keep mic processing active but suppress normal transcriptions during
    TTS and keep stop-phrase interruption available;
  - `full`: allow speech-triggered interruption during TTS, intended for AEC or
    headset scenarios;
  - `ptt`: push-to-talk tuned capture profile that behaves like `stop` while the
    assistant is speaking.
- The default mode is `wait` unless a later ADR changes it. That default is
  intentionally reliability-first rather than "most magical."
- Stop phrases are a first-class interruption path and must work without AEC.
- AEC remains an optional extra. True speech-triggered barge-in on speakers is
  not part of the default install contract.

## Consequences

### Positive

- The user-facing behavior is predictable and portable across machines.
- The package can offer interruption without pretending that all environments
  support real barge-in equally well.
- Optional AEC can improve the experience without becoming a hard dependency.

### Negative

- Users may need to choose between robustness and immediacy.
- `full` mode on speakers can still misbehave without AEC or a headset.
- Integrators must expose or document the mode choice instead of assuming one
  universal default.

### Neutral

- The interruption contract spans recognition, playback, REPL UX, and optional
  AEC, so changes often touch multiple layers.

## Enforcement

- Do not silently collapse `wait`, `stop`, `full`, and `ptt` into the same
  behavior.
- Do not claim true speech-triggered barge-in on speakers unless the active path
  actually supports it.
- Changes to default mode semantics, stop-phrase behavior, or AEC assumptions
  require an ADR update or superseding ADR.
- User-facing docs must describe the echo/self-interruption caveat for `full`
  mode.

## Validation

- `tests/test_adr0002_phase1.py`
- `tests/test_adr0002_phase2_optional_aec.py`
- `tests/test_full_mode_echo_gate.py`
- `tests/test_voice_recognizer_ptt_profile.py`

## Backlog links

- `docs/backlog/completed/015_adr0002_phase1_wait_mode_stop_phrase.md`
- `docs/backlog/completed/016_adr0002_phase2_optional_aec.md`

## Related

- [0001-local_assistant_out_of_box.md](./0001-local_assistant_out_of_box.md)
- [../architecture.md](../architecture.md)
- `abstractvoice/recognition.py`
- `abstractvoice/vm/core.py`
