## Task 057: the clone speaker check (056) streams continuously

**Date**: 2026-10-29  
**Status**: Completed (0.19.5)  
**Priority**: P1  

---

## Main goals

- Cloned speech plays without gaps while keeping 056's rule (one voice per reply, one regeneration, loud stop).

## Context / problem

Operator, after 0.19.4 reached the local Mac stack: speech from the gateway has short cuts every 2-4 s. 0.19.4 held
every generation's opening and then every 4 s block until it was scored, and released it at once.

Measured (isolated process, real weights, 0.6B Base MLX 8-bit, sisko, the incident reply's first two segments, ~80 s
of audio; a listener playing in real time from the first chunk received; "RTF" paced to stand in for a model
generating on the same GPU). Not measured on the operator's gateway: a stream request there creates a child run and
an artifact in the gateway data dir, which this investigation treats as read-only.

| code | RTF | first audio | gaps (max / p95) | largest lump (chunks at once) | provisional |
|---|---|---|---|---|---|
| 0.19.4 | 0.15 (idle) | 0.69 s | 0 / 0 | 6 | - |
| 0.19.5 | 0.15 (idle) | 1.40 s (opening regenerated) | 0 / 0 | 2 | 0 s |
| 0.19.4 | 0.6 | 1.45 s | 492 ms / 492 ms | 5 | - |
| 0.19.5 | 0.6 | 1.45 s | 0 / 0 | 1 | 3.8 s |
| 0.19.4 | 0.85 | 2.05 s | 1687 ms / 1687 ms | 5 | - |
| 0.19.5 | 0.85 | 7.37 s (opening regenerated) | 0 / 0 | 2 | 3.8 s |

The audio is bit-identical with and without the check (same seed), so 0.19.4 changed delivery timing only. At idle
speed 0.19.4 did not starve this simulated listener; it delivered in 4-5 s lumps, which a player that chains one
buffer per chunk on `onended` (the kit's `use_gateway_voice.ts`, `start_stream_segment`) has to absorb in bursts of
decoding. Whether those lumps are the operator's cuts at idle speed is not verified (no client-side timing exists).

## What changed

- `_Outbox` (one per reply): released audio goes out one chunk per generated chunk, plus whatever keeps the listener
  at least `LEAD_FLOOR_SECONDS` (1.5 s) ahead of real-time playback.
- `_checked_generation` is a delay line after the opening: chunk i is released once the 4 s window starting at it
  scores >= 0.93; a failing window raises before its first chunk plays. Only the reply's first opening is held
  unconditionally; a later opening is checked while the previous segment plays, and if the listener runs short a
  chunk goes out before its score (recorded as `provisional_s`) and is still scored: a miss then stops the reply
  (`mid_segment=True`). Verified audio queued before a stop is delivered first.
- Records: `audio_s` is the accepted generation's length; `provisional_s` added.

## Validation

- `tests/test_qwen3_tts_clone_stream_continuity.py` (virtual clock, weight-free):
  `test_a_long_reply_plays_without_gaps_at_any_realtime_factor[0.2|0.6|0.9]`,
  `test_delivery_follows_the_generation_cadence_not_held_blocks`,
  `test_at_ordinary_speed_every_chunk_is_verified_before_it_plays`,
  `test_a_stock_voice_segment_still_never_plays_and_the_reply_stops_naming_it`,
  `test_near_realtime_the_listener_is_fed_and_a_voice_change_still_stops_the_reply`.
  On the 0.19.4 engine 5 of 7 fail; without the provisional release 3 fail, without the one-per-chunk pacing 1, without
  the lead top-up 3. The 056 tests (`tests/test_qwen3_tts_clone_speaker_check.py`) stay green.
- Fast suite: 900 passed, 4 skipped.

## Follow-ups

- At RTF >= 0.6 the first ~10 s of a reply include provisional chunks (the delay line needs a lead it has not built yet).
- Kit: scheduling each decoded buffer ahead (`source.start(nextStartTime)`) instead of starting it from the previous
  buffer's `onended` would make playback gapless whatever the arrival pattern.
