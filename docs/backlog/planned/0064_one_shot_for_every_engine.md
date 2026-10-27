## Task 0064: `one_shot` is a framework guarantee, available on every engine

**Date**: 2026-10-27  
**Status**: Planned  
**Priority**: P1  
**Area**: TTS adapters, cloning engines, `TtsMixin.speak_to_bytes`, capability catalog, AbstractCore plugin `tts(one_shot=)`  
**Version**: today native on Qwen3-TTS only (0.18.3 declares it in the catalog)

---

## Main goals

- `one_shot=True` means the same thing on every engine: the whole text is rendered as one
  coherent take with one voice from the first word to the last, never cut and never silently
  segmented. It is a contract AbstractVoice owns, not a feature an engine happens to have.

## Secondary goals

- The capability catalog reports how each engine meets it (`native` when the engine renders the
  text in one generation, `emulated` when AbstractVoice composes it), with each engine's limit in
  `metadata.max_chars`, so callers never guess.

---

## Context / problem

`one_shot` exists because segmented synthesis samples the voice again for every segment, which
made the built-in voice references change voice mid-clip. Today only Qwen3-TTS honours it (one
generation, up to 1000 characters); every other engine raises `ValueError`. For the framework,
voice consistency across a take is a product property, so the decision is that every engine
offers it.

---

## Current code reality (2026-10-27, 0.18.3)

- `TtsMixin.speak_to_bytes(..., one_shot=True)` raises `ValueError` unless the adapter has
  `supports_one_shot`; only `abstractvoice/adapters/tts_qwen3_tts.py` sets it.
- Cloned speech: `abstractvoice/cloning/engine_qwen3_tts.py` has `supports_one_shot` and `_one_shot`;
  other clone engines do not.
- Catalog (`assets/voice_model_capabilities.json`): `one_shot` native on Qwen3-TTS `bytes` (per
  speaking checkpoint) and on Qwen3-TTS cloning `speak_bytes`; unsupported elsewhere.
- AbstractCore 2.30.0 forwards `one_shot` on `/v1/audio/speech`; a refusal answers 422.

---

## Constraints

- Never cut text and never pretend: an engine that cannot render a text as one take refuses it
  with the limit in the message (as Qwen3-TTS does beyond 1000 characters).
- Keep the public signature (`one_shot: bool`) unchanged.

---

## Research, options, and references

- **Option A**: per engine, the engine's own single-generation path where it exists (Supertonic and
  Piper render a whole utterance with fixed voice parameters; OmniVoice and other clone engines
  condition every chunk on the same reference or prompt), with each engine's real maximum length.
- **Option B**: for engines whose voice is deterministic per call (fixed style or speaker
  embedding), segmented synthesis already keeps one voice: report `emulated` and join the segments
  without gaps or fades.
- Measure per engine: speaker-embedding similarity across windows of a long take (the method used
  to validate the built-in voices) to prove "one voice".

---

## Decision

**Chosen approach**: Option A where an engine has a real single-generation path, Option B where
the voice is provably constant across segments; measured, never assumed.

---

## Dependencies

- **ADRs**: `docs/adr/0007_directed_speech_requests_and_planning_are_package_owned.md`
- **Backlog tasks**: related `docs/backlog/planned/0062_speed_ignored_in_bytes_synthesis_for_supertonic_and_piper.md`

---

## Implementation plan

- Audit each engine's single-take capability and limit.
- Implement `supports_one_shot` per adapter and clone engine; keep the refusal beyond each limit.
- Declare `one_shot` per engine in the catalog with `max_chars`.
- Weight-free tests per engine (no segmentation when `one_shot`), gated real-weights voice-stability checks.

---

## Success criteria

- `speak_to_bytes(text, one_shot=True)` succeeds on every local engine within its limit and refuses beyond it.
- The catalog's `one_shot` entry is `native` or `emulated` for every engine, with `max_chars`.
- A 30-second take keeps one voice (window similarity above the threshold used for the voice pack).

---

## Test plan

- `pytest -q`
- Gated per-engine stability measurement.

---

## Report (fill only when completed)

### Summary

### Validation
