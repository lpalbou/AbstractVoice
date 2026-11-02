## Task 058: tts_stream audio lines carry at most 2 s of audio

**Date**: 2026-11-02  
**Status**: Completed (0.19.5)  
**Priority**: P1  

---

## Main goals

- A client of the AbstractCore plugin's `tts_stream(...)` on a slow link starts playing a base-engine reply
  before a whole engine segment has downloaded.

## Context / problem

Base TTS engines (Supertonic, Piper, ...) hand `speak_to_audio_chunks()` a whole sentence group at once: up to about
16 s, 1.2 MB of 44.1 kHz WAV in one `audio` event. A client can play nothing of an event before it has all of it, so
on a slow link (a watch over Wi-Fi/LTE, 2026-11-01, "Voice unavailable") the first audio waited for the whole
segment. Qwen3-TTS clone chunks are already about 1 s.

## Implementation

- `abstractvoice/integrations/abstractcore_plugin.py`: `STREAM_LINE_MAX_SECONDS = 2.0` and
  `_split_audio_for_stream(audio, sample_rate)`; `tts_stream` encodes each slice as its own standalone WAV `audio`
  event with consecutive `sequence` numbers. A chunk within the cap is one slice (clone chunks unchanged).
- `done.chunks` counts the events sent; `done.metrics` keeps the engine's own counts.

## Evidence

- `tests/test_abstractcore_plugin.py::test_voice_capability_tts_stream_slices_long_segments_into_short_lines`:
  10 s at 24 kHz leaves as 5 events of at most 2.05 s whose PCM concatenates byte-identically to the
  segment's own PCM; `done.chunks == 5`, engine metrics untouched.
- `tests/test_abstractcore_plugin.py::test_voice_capability_tts_stream_leaves_short_chunks_whole`: 1 s stays one
  slice; 100000 samples at 24 kHz split 48000/48000/4000; empty stays one empty slice.
- Mutation: replacing `_split_audio_for_stream` with a pass-through turns the first test red.
- Default suite 925 passed, 3 skipped (release 0.19.5, 2026-11-02).

## Docs

- `docs/api.md` (`tts_stream` event contract), `docs/architecture.md` (plugin streaming diagram), `llms.txt`,
  `llms-full.txt`, `CHANGELOG.md` 0.19.5.
