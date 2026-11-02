# ADR 0009: Streamed speech resumes at text-segment boundaries

Status: Accepted.

## Context

Clients read long replies aloud over unreliable links (a watch over Wi-Fi or LTE): they
request speech for a piece of text through the AbstractCore plugin's `tts_stream(...)`
(`POST /runs/{id}/voice/tts/stream` on AbstractGateway) and play the audio as it arrives.
When a stream breaks part-way, the client must request the rest.

Skipping the seconds of audio already received does not resume exactly. Synthesis is not
deterministic: Qwen3-TTS samples every generation (temperature 0.9, no seed), the speaker
check can regenerate a whole generation (ADR 0004, backlog 056), and engines split text
differently (Supertonic and Piper into sentence groups of at most 240 characters, the
Qwen3-TTS clone into generations of up to 800 characters on MLX). A second synthesis of
the same text has different timing, so a time offset lands a few words early or late:
the listener hears words twice or misses them.

What is exact is the text. Every segment an engine synthesizes is one independent
generation of a known slice of the text, and its audio speaks exactly that slice.

## Decision

- A streamed synthesis is a sequence of text segments. The segment-aware iterators
  (`VoiceManager.speak_to_audio_segments`, `VoiceCloner.speak_to_audio_segments`, and
  `infer_to_audio_segments` on engines that report their generations) yield
  `SegmentStart`, that segment's `(audio, sample_rate)` chunks, then `SegmentEnd`, in
  delivery order (`abstractvoice.tts.segments`). The chunk iterators are the same streams
  without the markers.
- A segment is the engine's own unit of generation: a base engine's sentence group, one
  Qwen3-TTS clone generation (a regenerated take is still that segment; only the accepted
  take is delivered). A native-streaming base engine, and a clone engine that does not
  implement `infer_to_audio_segments`, make the whole text segment 0. Coarser segments are
  allowed; a segment that does not speak exactly its text is not.
- The voice manager locates every segment in the caller's text: `text_start` / `text_end`
  are code-point offsets `[start, end)` into the text as the caller sent it. The Markdown
  markers the sanitizer removes and the whitespace the segmenters collapse are mapped
  back (`SegmentLocator`); a segment that is not the next slice of the text raises.
- `tts_stream` puts `segment`, `text_start` and `text_end` on every `audio` event (all
  2 s slices of one segment share them) and writes one
  `{"type": "segment_end", "segment", "text_start", "text_end"}` event after a segment's
  last audio. `done` carries `segments`, the number of segments ended.
- The client retry rule: after a break, request the text from `text_end` of the last
  `segment_end` received (from the start when there is none), and discard the audio of
  the segment that had no `segment_end`. Offsets in the retried stream index the retried
  text.
- AbstractVoice is the authoritative producer. Hosts between the plugin and the client
  (AbstractRuntime, AbstractGateway) forward the text and these fields unchanged; they do
  not strip the text, renumber segments, or derive their own offsets.

## Consequences

### Positive

- Resume is exact for every engine, whatever it samples: no word is repeated or missed
  at the seam.
- Clients need no timing model of the engine and no text matching; the offsets index
  their own text.
- The fields are additive: existing clients that ignore them keep working.

### Negative

- A segment is the resume unit. Audio of a partially received segment is discarded, so a
  break in a long segment repeats it from its start. On the MLX backend a clone reply of
  up to 800 characters is one generation (ADR 0008), so a client that sends 600-character
  pieces restarts the piece it was in. Finer resume would need finer generations, which
  trades against one voice per reply.
- An emphasis marker that spans two segments (`*first. second.*`) leaves its closing
  marker in the retried text. The contract is exact for plain text; clients that resume
  send plain text.

### Neutral

- Base and clone engines keep their own segmentation; the contract reports it rather than
  imposing one.

## Enforcement

- A new engine either implements `infer_to_audio_segments` with segments in delivery order
  or is treated as one segment; it never reports a segment boundary its audio does not
  respect.
- A new text transform before synthesis must only delete characters (or carry its own
  index map into `SegmentLocator`); otherwise the locator raises.
- Hosts that relay `tts_stream` pass the request text through unaltered and forward unknown
  event fields; AbstractGateway's voice stream test checks this.

## Validation

- `tests/test_tts_stream_segments.py`: spans are exact slices of French and English text
  with punctuation, numbers and emoji; Markdown markers map back; the remainder after any
  segment speaks exactly the rest; base, native and clone paths mark segments in delivery
  order (through a speaker-check regeneration and with the check off); a cancelled
  segment never ends; the plugin's lines carry the fields on every slice; the client retry
  rule, applied after every line of a stream, loses and repeats nothing.
- `tests/test_voice_cloner_engine_dispatch.py`: an engine without
  `infer_to_audio_segments` is one segment.
- AbstractGateway `tests/test_voice_stream_segments.py`: the text reaches the engine as
  sent and the fields reach the client unchanged.

Related: backlog 0066, ADR 0004, ADR 0008.
