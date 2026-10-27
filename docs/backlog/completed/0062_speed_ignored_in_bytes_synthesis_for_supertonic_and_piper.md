## Task 0062: Speed is ignored when Supertonic and Piper synthesize to bytes

**Date**: 2026-10-26  
**Status**: Completed  
**Completed**: 2026-10-27  
**Priority**: P2  
**Area**: TTS adapters (`abstractvoice/adapters/tts_supertonic.py`, `abstractvoice/adapters/tts_piper.py`), `TtsMixin.speak_to_bytes`  
**Version**: found in 0.18.2; deferred to the package rebuild

---

## Main goals

- A speed set with `VoiceManager.set_speed(...)` or passed per call (`speed=` on the AbstractCore plugin `tts()`, `speed` on `POST /v1/audio/speech`) changes the audio returned by `speak_to_bytes()` / `speak_to_file()` for Supertonic and Piper, as it already does for live playback.

## Secondary goals

- The capability catalog (`compatibility_catalog()`, `tts_capabilities`) reports speed for the `bytes` surface truthfully for every engine, so a caller never gets a silently ignored control.

---

## Context / problem

`speak_to_bytes()` builds a speech request whose `speed` is the manager's speed, then calls
`tts_adapter.synthesize_to_bytes_with_voice(..., speed=effective_speed, ...)` when the speed is
not 1.0. Both adapters accept that call but drop `speed` into `**_kwargs`:

- Supertonic synthesizes with its own `self._speed` (1.05 by default), whatever the request says.
- Piper ignores it entirely.

So the buffered/file surface (used by the AbstractCore plugin and `/v1/audio/speech`) returns
audio at the default pace while the caller believes the speed was applied. Live playback is not
affected. Reported to the maintainer during the 0.18 cycle; deferred because the package is due
for a rebuild and the fix touches adapter contracts.

---

## Current code reality (2026-10-26, 0.18.2)

- `abstractvoice/vm/tts_mixin.py` `speak_to_bytes`: `request = self._build_speech_request(..., speed=float(getattr(self, "speed", 1.0) or 1.0))`; `effective_speed` is dropped only when the compatibility catalog says `unsupported` for `surface="bytes", feature="speed"`; otherwise forwarded as `speed=` to `synthesize_to_bytes_with_voice`.
- `abstractvoice/adapters/tts_supertonic.py`: `synthesize_to_bytes_with_voice(self, text, *, format="wav", voice=None, **_kwargs)` never reads `speed`; `synthesize_to_bytes` uses `self._speed`; `_synthesize_with_speed(text, speed)` already exists and takes an explicit speed.
- `abstractvoice/adapters/tts_piper.py`: `synthesize_to_bytes_with_voice(self, text, *, format="wav", voice=None, **_kwargs)` never reads `speed`.

---

## Constraints

- Keep the public `VoiceManager` contract and the plugin/server request shape unchanged.
- Valid range stays 0.5-2.0 (`set_speed` validation).
- No new heavy dependency; Piper speed should use Piper's own length scale, not post-hoc resampling that shifts pitch.

---

## Research, options, and references

- **Option A**: honour `speed` in each adapter's `synthesize_to_bytes_with_voice` (Supertonic via `_synthesize_with_speed`; Piper via its `length_scale` synthesis option, `length_scale = 1 / speed`).
  - Fixes the root cause where the knowledge lives; the catalog can then claim `native`.
- **Option B**: mark speed `unsupported` on the bytes surface for these engines.
  - Honest but removes a feature that works in playback.
- **Option C**: time-stretch the produced audio in the mixin.
  - Engine-independent, but adds a DSP dependency and artefacts; rejected.

---

## Decision

**Chosen approach**: Option A, plus a catalog check that fails if any engine advertises speed on
a surface whose adapter drops it.

**Why**:
- The engines already support speed natively; only the bytes path loses it.
- A silently ignored control is worse than an unsupported one.

---

## Dependencies

- **ADRs**:
  - `docs/adr/0007_directed_speech_requests_and_planning_are_package_owned.md` (request fields and per-engine capability truth, including explicit degradation)
- **Backlog tasks**:
  - Related: `docs/backlog/planned/035_piper_voice_profiles_and_voice_selection.md`

---

## Implementation plan

- Read `speed` in both adapters' `synthesize_to_bytes_with_voice` (and Supertonic's `synthesize_to_bytes` from the request, not only `self._speed`).
- Make `VoiceManager.set_speed` and the per-call speed reach the same code path.
- Add a weight-free test per engine that the speed reaches the synthesis call, and a gated real-weights test that a 1.5x render is measurably shorter than 1.0x.
- Align the compatibility catalog's `speed` support on the `bytes` surface with the adapters.

---

## Success criteria

- `speak_to_bytes` at speed 1.5 yields audio about 1/1.5 as long as at 1.0 for Supertonic and Piper (within 10%).
- `POST /v1/audio/speech` with `speed` changes the duration for both engines through AbstractCore.
- No engine advertises `speed` on a surface that ignores it.

---

## Test plan

- `pytest -q`
- Gated: render one sentence at 0.75, 1.0, 1.5 with each engine and compare durations.

---

## Report (fill only when completed)

### Summary

Shipped in 0.18.4, together with Qwen3-TTS speed. Supertonic honours `speed` natively on the bytes path
(`_synthesize_with_speed`, the playback rule: 1.0 keeps the engine's default pace). Piper, Qwen3-TTS
(preset speakers) and Qwen3-TTS clones apply the existing pitch-preserving time-stretch
(`apply_speed_without_pitch_change`: librosa when installed, else WSOLA) after generation, as Chroma already
did. The catalog declares `speed` native (Supertonic) or emulated (Piper, Qwen3-TTS bytes, Qwen3-TTS clone
`speak_bytes`); Qwen3-TTS live playback stays unsupported. Two root causes surfaced and were fixed:
`set_speed` refused any speed when live playback could not apply it (so the AbstractCore plugin, which applies
a per-call speed through `set_speed`, dropped it for Qwen3-TTS); it now keeps a speed that any path can apply.
And a model record in the catalog reset every feature it did not name to unsupported, which dropped per-call
preset voices and quality presets for Qwen3-TTS on the bytes path; model records now inherit their provider's
defaults. Option A of the decision was followed; Piper uses the time-stretch rather than `length_scale`
(simpler, same contract), which can be revisited.

### Validation

- Tests: `tests/test_speed_on_bytes_path.py` (each path measured on its output duration; catalog levels;
  model inheritance). Full suite 844 passed; torch-free CI mirror 689 passed.
- Real weights, `speak_to_bytes` at 0.8 / 1.25 (expected x1.25 / x0.80): Supertonic x1.31 / x0.84, Piper
  x1.20 / x0.79, Qwen3-TTS preset x1.37 / x0.79, Qwen3-TTS built-in clone x1.25 / x0.77; Whisper word error
  0.00 at 0.8, at most 0.13 at 1.25. Through the AbstractCore plugin (per-call `speed`, built-in clone, median of
  three takes): x0.85 at 1.25, x1.37 at 0.8.
