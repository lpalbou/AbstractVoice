## Task 056: a Qwen3-TTS cloned reply is in ONE voice, or it fails loudly

**Date**: 2026-10-29  
**Status**: Completed (0.19.4)  
**Priority**: P1  

---

## Main goals

- One cloned reply is spoken in one voice. A generation that comes out in another voice is regenerated once; if it
  still does not carry the clone, synthesis stops and names the text segment. Never a silent speaker switch.

---

## Context / problem

Operator report, 2026-10-29 ~03:12-03:15 CET: an AbstractCode automation's reply (8 kB of text), read aloud through
the gateway in the cloned voice "sisko" (`qwen3-tts`, `Qwen/Qwen3-TTS-12Hz-0.6B-Base`), was sisko for the first part,
a stock-sounding male voice for the "middle paragraphs", and sisko again from "It's an attacker that walks in through
your vendor".

Evidence (read-only, gateway data dir):

- One artifact holds the reply (`df965b6d12e6e9d42c70b4e6491bc5d5`, child run `7981cef4-…`, one attempt, terminal
  event `ok`, 452 wav-segment chunks, `engine: clone`, `clone_engine: qwen3-tts`, `voice_id` sisko). One request, one
  voice, one model, no retry and no error.
- Code path: plugin `tts_stream` → `vm.speak_to_audio_chunks(voice=sisko)` → `VoiceCloner.speak_to_audio_chunks` →
  `Qwen3TTSVoiceCloningEngine.infer_to_audio_chunks`. The MLX backend (auto on Apple silicon, 8-bit) splits the
  sanitized text (8032 chars) into 11 segments of at most 800 chars (`NATIVE_STREAM_MAX_CHARS`) and runs ONE
  `stream_clone` generation per segment with the SAME clone prompt object (built once per reply). No audio cache
  exists anywhere on the path (abstractvoice, AbstractRuntime `stream_voice`, gateway stream route).
- The artifact WAV was aligned to those 11 segments with Whisper word timestamps (large-v3-turbo, 1269 of 1338 words
  matched) and scored in 3 s windows with the 0.6B Base checkpoint's own speaker encoder against sisko's reference
  clip. Segment 8 (293.5-328.5 s, "The question for your team is no longer…" to "The agent is no longer a tool.")
  scores 0.930 against the reference and 0.958 against the reply's own sisko audio; every other segment scores
  0.963-0.983 and 0.982-0.993. The change is exactly at the segment boundaries, and segment 9 starts with "It's an
  attacker that walks in". Other speakers in the operator's voice library score 0.87-0.944 against the same reference.

| seg | audio (s) | chars | first words | first 4 s vs ref | mean vs ref | mean vs reply's sisko | voice |
|---|---|---|---|---|---|---|---|
| 0 | 0.0-41.0 | 786 | Here's your digest, written to be… | 0.979 | 0.977 | 0.993 | sisko |
| 1 | 41.0-78.6 | 750 | The agents walked in through the… | 0.986 | 0.981 | 0.990 | sisko |
| 2 | 78.6-116.6 | 796 | Now, the other half of the… | 0.990 | 0.982 | 0.990 | sisko |
| 3 | 116.6-154.6 | 791 | It's going first to selected cyber… | 0.987 | 0.976 | 0.988 | sisko |
| 4 | 154.6-192.3 | 703 | Researchers at Huntress documented the ConnectWise… | 0.969 | 0.963 | 0.982 | sisko |
| 5 | 192.3-227.0 | 678 | Microsoft recorded attempted sign-ins against more… | 0.987 | 0.983 | 0.989 | sisko |
| 6 | 227.0-262.4 | 755 | The big one: the high-risk regime… | 0.986 | 0.982 | 0.989 | sisko |
| 7 | 262.4-293.5 | 699 | First, if you run systems, the… | 0.981 | 0.977 | 0.991 | sisko |
| 8 | 293.5-328.5 | 745 | The question for your team is… | 0.941 | 0.930 | 0.958 | **another voice** |
| 9 | 328.5-357.5 | 596 | It's an attacker that walks in… | 0.969 | 0.979 | 0.992 | sisko |
| 10 | 357.5-413.5 | 702 | Sources for this edition: TechRadar… | 0.968 | 0.976 | 0.989 | sisko |

Every row: prompt = sisko ICL (ref_0.wav, 3.9 s, manual transcript), model `Qwen3-TTS-12Hz-0.6B-Base`, backend MLX
8-bit, native stream; no audio cache exists.

Cause: the model, not a code path. Real weights, isolated process, same settings: segment 8's exact text came out
in sisko 6 of 6 times, while segment 0's exact text once came out in another voice (0.932). Over 70+ generations of
five clones, two (sisko, reference 3.9 s; hal9000-qwen2, 6.4 s) open a generation in a stock voice in roughly 5-15%
of generations (sisko 0 of 12 to 2 of 8 per batch, hal9000-qwen2 up to 3 of 7); three did not in 8 tries each
(hal9000-qwen 3.5 s, pavel 2.85 s, pascale-qwen 20.5 s). Reference length does not predict it on this sample.
Sampling at the "low" quality preset (temperature 0.7) did not prevent it. The
generation reports success, so nothing upstream could notice.

Ruled out: (a) an audio cache keyed without the voice (none exists; segment 8's text is unique to this reply);
(c) routing to another engine, model or backend (one engine, one runtime, one prompt object per reply; the chunk
count, 452, matches 11 MLX native streams); (b) a swallowed per-chunk clone failure, a dropped prompt or a stock
speaker fallback (no such path; the prompt is rebuilt for every generation from the same arrays, and segment 8's
text reproduces fine).

## What changed

- `Qwen3TTSRuntime.speaker_similarity(audio, clone_prompt=, sample_rate=)`: cosine between the speaker of `audio`
  and the prompt's speaker embedding, with the same Base speaker encoder (MLX and PyTorch). Lock-free: it is called
  while a clone stream holding the generation lock is suspended between chunks.
- `Qwen3TTSVoiceCloningEngine` checks every generation (chunked, native stream, buffered and one-shot):
  - the opening is held back and scored: released at 2 s when >= 0.975, else decided at 4 s against 0.95;
  - a miss closes that generation (releasing the model) and runs it again once; a second miss raises
    `ClonedVoiceNotAppliedError` (segment index and count, text, scores, threshold);
  - after the opening, each 4 s block is scored before release; below 0.93 raises (`mid_segment=True`), since the
    start of the segment was already heard;
  - audio under 1 s is too short to judge and passes.
- Per-segment records (engine `last_segments`, `VoiceCloner.last_segment_records()`, INFO log, stream metrics
  `segments`, `ClonedVoiceNotAppliedError.segments`).
- `reference_seconds` on clone records (store list/get; plugin `tags.reference_seconds`).
- `ABSTRACTVOICE_QWEN3_TTS_SPEAKER_CHECK=0` disables the check; `ABSTRACTVOICE_QWEN3_TTS_MIN_SPEAKER_SIMILARITY`
  retunes the opening threshold. Explicit constructor arguments win.
- Cost: first audio of each generation waits for the check, ~0.5 s (2 s release) or ~1.1 s (4 s) at the MLX
  backend's real-time factor; later generations are checked while the previous one plays.

Calibration (cosine vs the reference embedding, MLX 8-bit): clone openings >= 0.956 at 4 s (n=43 + the 10 sisko
segments above), later clone blocks >= 0.948 (n=182); stock-voice openings <= 0.941 at 4 s and none >= 0.975 at 2 s;
stock-voice and runaway blocks <= 0.92.

## Validation

- `tests/test_qwen3_tts_clone_speaker_check.py` (weight-free): regenerate-once and fail-loudly on both the MLX native
  stream and the PyTorch segment path, the held opening (2 s early release, 4 s decision), a voice change part-way,
  buffered and one-shot synthesis, too-short audio, the environment overrides, `speaker_similarity` itself, and a
  guard that no per-text audio cache exists. Red with the check bypassed (7 of 9 engine tests fail), with
  `SPEAKER_RETRIES = 0` (5 fail), without the mid-segment block check (1 fails) and without the 2 s early release
  (1 fails).
- `test_every_segment_leaves_a_traceable_record_and_metrics_carry_them` and
  `test_clone_records_carry_the_reference_length`: red without the metrics line, without the exception's records,
  without the store field and without the listing tag, respectively.
- Fast suite (`pytest -m "not integration and not model_download"`): 893 passed, 4 skipped.
- Real weights (0.6B Base, MLX 8-bit, isolated process, voice store read-only), check ON:
  - the exact reply, twice, all 11 segments: no regeneration, no failure, every scored block 0.946-0.989
    (no false alarm on the incident text);
  - 28 single-segment replies (420 chars; sisko and hal9000-qwen2, 14 each): hal9000-qwen2 0 interventions; sisko 3:
    two openings in another voice (0.948 and 0.623 at 4 s) regenerated, the regenerations delivered at 0.943 and
    0.969 over the whole; one voice change part-way (0.977 → 0.886 after 30 s, past where the text should have
    ended: a runaway tail) stopped loudly as `ClonedVoiceNotAppliedError`, segment named. First audio: 0.5-1.0 s
    (1.3-2.7 s with a regeneration), against 0.14 s before.

## Not done

- Carrying the previous verified segment forward into the next segment's ICL prompt (to lower the miss rate before
  detecting it) was proposed but not measured; it is not in this change. The proposal rested on "short references
  drift", which the data above contradicts.
- No short-reference warning: reference length does not predict the misses on this sample. Only the length is
  exposed (`reference_seconds`).

## Follow-ups

- The opening threshold sits close to the clone's own lowest opening scores (0.95 vs 0.956). A voice
  whose clone scores lower against its own reference (noisy clip) would fail loudly; the environment override is the
  escape hatch. A sturdier anchor is the clone's own synthesized audio (genuine vs genuine >= 0.989, stock vs genuine
  <= 0.952) rather than the reference recording.
- A regeneration that passes its opening can still sit in the band between the clone's lowest blocks (0.948) and the
  block stop (0.93): sisko run 7 above was delivered at 0.936-0.953. Not verified by ear.
- The runtime's stream error event (`abstractruntime` `run_facade.stream_voice`, the `except Exception` branch near
  line 722) records `str(exc)` only; to put the records of a failed reply in the child run too, it would add
  `getattr(exc, "segments", None)` to the event. Not changed here (another package).
- One calibration generation (hal9000-qwen2, temperature 0.7) missed its EOS and ran to the runaway cap (160 s for
  420 chars); the block check catches its tail (0.88) but the cap itself is generous.
- Which voices are prone is not explained: reference length does not predict it (two of the three clones that never
  missed have the shortest references).
