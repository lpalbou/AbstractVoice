## Task 0066: tts_stream names each audio line's text segment, for an exact resume

**Date**: 2026-11-02  
**Status**: Completed (unreleased)  
**Priority**: P1  

---

## Main goals

- A client whose speech stream broke resumes with no repeated and no missing word, for every engine, including
  the sampled Qwen3-TTS clone.

## Secondary goals

- Additive, backward-compatible stream fields; the gateway forwards them unchanged.

## Context / problem

The watch and iPhone read long answers in ~600-character pieces (one `tts_stream` request per piece). After a break
they retried the piece and skipped the seconds of audio already received. For the Qwen3-TTS clone that is only
approximate: generations are sampled (temperature 0.9, no seed), the speaker check can regenerate a whole generation,
and the retry segments differently, so the seam repeated or dropped 1-6 words, sometimes a sentence. Audio events
carried only `sequence`, format and sample rate.

## Implementation

- `abstractvoice/tts/segments.py`: `SegmentStart` / `SegmentEnd` markers between the `(audio, sr)` chunks,
  `audio_only`, `single_segment`, and `SegmentLocator` (segment text -> code-point span of the caller's text, through
  the Markdown sanitizer's index map and the whitespace collapse; raises if a segment is not the next slice).
- `abstractvoice/text_sanitize.py`: `sanitize_markdown_with_index` (the sanitizer is its first element, so both
  cannot diverge).
- `VoiceManager.speak_to_audio_segments`: base sentence groups, native single generation, clone generations; spans
  filled by the locator; `speak_to_audio_chunks` is its audio. `VoiceCloner.speak_to_audio_segments`: the engine's
  `infer_to_audio_segments`, else the whole text as segment 0.
- Qwen3-TTS clone `infer_to_audio_segments`: markers travel through the reply's `_Outbox` in delivery order (they
  carry no audio, the pacing ignores them; a segment's end leaves at once when its audio has all gone out); with the
  speaker check off, chunks go through the outbox too so markers stay in order.
- Plugin `tts_stream`: `segment`, `text_start`, `text_end` on every audio line, a `segment_end` line after each
  segment's last slice, `done.segments`.
- Durable rule: ADR 0009.

## Evidence

- `tests/test_tts_stream_segments.py` (19 tests): exact spans on French and English text with punctuation, numbers
  and emoji (ZWJ family, skin tone); Markdown mapping; remainder after any segment == the rest; base, native and clone
  (MLX native and torch, through a regeneration, speaker check off) in delivery order; cancelled segment never ends;
  plugin lines; the client retry rule applied after every line loses and repeats nothing.
- `tests/test_voice_cloner_engine_dispatch.py`: an engine without `infer_to_audio_segments` is segment 0.
- Mutations turned red: end marker yielded directly instead of through the outbox; no marker extension in the
  locator (end and start); speaker-check-off chunks bypassing the outbox; base path without `SegmentEnd`.
- Default suite: 943 passed, 4 skipped.
- AbstractGateway `tests/test_voice_stream_segments.py`: the text reaches the engine as sent, fields pass through,
  the audit counts `segments`.

## Docs

- `docs/api.md` (`speak_to_audio_segments`, `tts_stream` segment fields and the retry rule), `docs/architecture.md`
  (plugin streaming diagram), ADR 0009, `README.md`, `llms.txt`, `llms-full.txt`, `CHANGELOG.md` Unreleased.

## Follow-ups

- Apple clients (`SpeechEvent` in abstractassistant's `AbstractGatewayClient`) decode the fields and apply the retry
  rule.
- Resume granularity on MLX is one generation (up to 800 characters per reply, ADR 0008): finer resume would need
  shorter generations, a trade against one voice per reply.
