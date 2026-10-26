## Task 0063: Voices cloned from a recording record no language

**Date**: 2026-10-26  
**Status**: Planned  
**Priority**: P2  
**Area**: cloning (`abstractvoice/cloning/`), AbstractCore plugin `clone()` and voice navigation (`_with_voice_navigation`), AbstractCore `POST /v1/voice/clone`  
**Version**: found in 0.18.2

---

## Main goals

- A voice cloned from a recording stores the language it was recorded in, so voice navigation reports its native `language` and lists it among the native speakers when filtering by that language, as it already does for built-in and designed voices.

## Secondary goals

- One documented way to state the language at clone time, honoured for every audio input form (bytes, path, artifact reference) and forwarded by AbstractCore's `POST /v1/voice/clone`.

---

## Context / problem

Built-in voices carry `meta.language_code` (voice pack) and designed voices carry it
(`design_voice` writes it). A voice cloned from a recording carries none, so the plugin's
navigation record has `language: null`. It still appears under `language=` filters for every
language its engine speaks, but never among the native speakers listed first: a French
recording cloned for French use is ranked after every native French voice.

---

## Reproduction (exact experiment, 2026-10-26, abstractvoice 0.18.2)

Environment: `abstractvoice[qwen3-tts]` installed (creating a Qwen3-TTS clone stores the reference
and does not run the model). Run from a directory outside the repository:

```python
import io, wave
import numpy as np
from abstractvoice.integrations.abstractcore_plugin import _VoiceCapability

class Owner:
    config = {"voice_allow_downloads": False}

cap = _VoiceCapability(Owner())
buf = io.BytesIO()
with wave.open(buf, "wb") as w:  # 5 s of tone stands in for a recording
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000)
    w.writeframes((np.sin(np.arange(24000 * 5) / 10) * 3000).astype("int16").tobytes())
vid = cap.clone(buf.getvalue(), provider="qwen3-tts", name="repro-fr", reference_text="Bonjour et bienvenue à tous.")
try:
    rec = cap.get_cloned_voice(vid)
    print("stored meta:", {k: rec.get("meta", {}).get(k) for k in ("language", "language_code", "designed")})
    fr = cap.list_cloned_voices(provider="qwen3-tts", language="fr")
    entry = next(v for v in fr if v["voice_id"] == vid)
    print("listed language:", entry["language"], "| speakable languages:", len(entry["languages"]))
    print("position among fr clones:", [v["voice_id"] for v in fr].index(vid) + 1, "of", len(fr),
          "| native fr voices listed first:", sum(1 for v in fr if v["language"] == "fr"))
finally:
    cap.delete_cloned_voice(vid)
```

Observed output (machine with the 70 built-in voices):

```text
stored meta: {'language': None, 'language_code': None, 'designed': None}
listed language: None | speakable languages: 10
position among fr clones: 12 of 76 | native fr voices listed first: 11
```

Expected after the fix: `language_code` is `fr`, `listed language: fr`, and the clone ranks among
the native French voices.

Over HTTP (AbstractCore 2.30.0): `POST /v1/voice/clone` with a French recording, then
`GET /v1/voice/clones?language=fr` shows the new voice with `"language": null`; the form has no
field to state the language.

---

## Current code reality (2026-10-26, 0.18.2)

- `_VoiceCapability.clone(audio, *, ..., metadata=None, ...)` passes `metadata` as the stored `meta`
  only when `audio` is bytes (`vm.clone_voice_from_wav_bytes(..., meta=clone_meta)`); the path and
  artifact-dict branches call `vm.clone_voice(...)` without it, so the same input in another form
  is stored differently. `metadata` is not documented as the place for a language.
- `VoiceManager.clone_voice(...)` has no `meta` / language parameter; `clone_voice_from_wav_bytes`
  has `meta`.
- `_with_voice_navigation` reads the native language from `tags.language_code` or
  `params.meta.language_code` only.
- When `reference_text` is omitted, the cloner transcribes the reference with Whisper, which detects
  the language, and discards it.
- AbstractCore `POST /v1/voice/clone` has no `language` form field.

---

## Constraints

- Keep `clone(...)` / `clone_voice(...)` backward compatible (the new argument is optional).
- Do not guess silently: an inferred language must say it was inferred (`language_source`).
- Keep creation weight-free when `reference_text` is given (no STT run just to infer a language).

---

## Research, options, and references

- **Option A**: explicit `language=` on `VoiceManager.clone_voice*`, the plugin `clone()` and the
  AbstractCore form, stored as `meta.language_code` for every input form.
  - Simple, exact, no inference.
- **Option B**: infer it: from Whisper's detected language when the reference is auto-transcribed,
  else from `reference_text` with the text language identifier already used for Supertonic
  (`py3langid`).
  - Works for callers that state nothing; short texts can be misidentified.
- **Option C**: A, falling back to B, recording `meta.language_source` (`request` | `stt` | `text`).

---

## Decision

**Chosen approach**: Option C.

**Why**:
- The caller knows best (A), but most callers will not say it (B covers them).
- `language_source` keeps an inferred value from being mistaken for a stated one.

---

## Dependencies

- **ADRs**:
  - `docs/adr/0003_cloning_reference_text_fallback.md` (reference transcript handling)
  - `docs/adr/0007_directed_speech_requests_and_planning_are_package_owned.md`
- **Backlog tasks**:
  - Related: `docs/backlog/planned/0062_speed_ignored_in_bytes_synthesis_for_supertonic_and_piper.md` (same rebuild)
- **Cross-package**: AbstractCore `POST /v1/voice/clone` gains an optional `language` field forwarded to the plugin.

---

## Implementation plan

- Add `language` to `VoiceManager.clone_voice`, `clone_voice_from_wav_bytes` and the plugin `clone()`; route `metadata` to the store for every audio input form.
- Keep Whisper's detected language when the cloner transcribes the reference; otherwise identify it from `reference_text`.
- Store `meta.language_code` and `meta.language_source`.
- Turn the reproduction above into a weight-free test (it already runs without model weights) asserting `language_code == "fr"` and native-first ranking.
- Coordinate the AbstractCore form field.

---

## Success criteria

- The reproduction prints `language_code` `fr`, `listed language: fr`, and the clone ranks among the native French voices.
- Bytes, path and artifact inputs store the same metadata.
- `GET /v1/voice/clones?language=fr` through AbstractCore shows the clone with `"language": "fr"`.

---

## Test plan

- `pytest -q`
- The reproduction script above, before and after.

---

## Report (fill only when completed)

### Summary

### Validation
