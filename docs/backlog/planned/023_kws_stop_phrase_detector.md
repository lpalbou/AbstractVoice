## Task 023: Keyword spotting (KWS) for reliable stop phrase

**Date**: 2026-01-23  
**Status**: Planned  
**Priority**: P0  

---

## Goal

Make stop-phrase interruption reliable **without** requiring full ASR during playback.

Target behavior:
- User says **"ok stop"** (or "stop") while TTS plays on speakers
- TTS stops within ~200–500ms
- No false triggers from the assistant’s own audio

---

## Context / problem

Using Whisper/faster-whisper as a stop-word detector is fundamentally brittle under echo:
- it is slower than dedicated KWS
- it can hallucinate stop words (especially with hotword bias)
- it often requires end-of-utterance to trigger transcription

---

## Licensing constraint (critical)

We must only ship **MIT/Apache/BSD** licensed components.

`openwakeword` (Apache-2.0) is acceptable **as a library**, but its hosted pretrained
wakeword models are commonly distributed under **non-commercial** terms (varies per model),
which is **not acceptable** for AbstractVoice core.

Therefore, we can only use `openwakeword` if we:
- train our own KWS model(s) using a permissively-licensed dataset, and
- ship those weights under a permissive license inside this repo (or via a permissive release artifact).

---

## Proposed approach

- Add a small KWS module `abstractvoice/kws/` using `openwakeword` runtime.
- Train a dedicated model for:
  - `"stop"`
  - `"ok stop"` (and/or `"okay stop"`)
- Run it continuously while TTS is playing in STOP mode.
- Gate triggers with:
  - a short debounce window
  - optional far-end correlation / AEC integration when available

---

## Dependencies

- `openwakeword` runtime (Apache-2.0)
- A permissively licensed dataset suitable for keyword spotting (to be researched and cited)

---

## Success criteria

- Stop phrase works reliably on laptop speakers (macOS/Windows/Linux)
- No surprise downloads in REPL
- No non-commercial model weights in repo

---

## Test plan

- Unit tests: signal-level synthetic + mocked KWS output
- Manual REPL: play long TTS, say stop phrase at varying volumes/distances

