## Task 0072: The one-shot TTS command does not fetch Supertonic on first use

**Date**: 2026-11-10  
**Status**: Proposed  
**Priority**: P2

---

## Main goal

`abstractvoice --tts-engine supertonic --prompt "Hello" --output hello.wav` works on a fresh install
without a separate prefetch step.

## Context / problem statement

Found while verifying the standalone installer on a fresh Ubuntu host (2026-11-10): right after
`abstractvoice[light]` is installed, the one-shot command fails with "Supertonic 3 artifacts are not
available locally" instead of downloading them, although the library path (`VoiceManager`,
`allow_downloads=True`) downloads on first use. The installer's closing hint works around it by
running `abstractvoice-prefetch --supertonic` first.

## Decision

Proposed: the one-shot CLI passes `allow_downloads=True` (the library default) unless `--offline`
is given, and says what it downloads and where. Same check for `--tts-engine piper` and the STT
one-shot path.

## Success criteria

- A fresh install's first one-shot command downloads and speaks; `--offline` refuses with the
  current message. Test with the Supertonic cache dir pointed at an empty scratch folder.
