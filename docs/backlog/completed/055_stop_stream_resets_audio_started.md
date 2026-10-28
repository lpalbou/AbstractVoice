## Task 055: `stop_stream()` resets the player's "speech started" state

**Date**: 2026-10-28  
**Status**: Completed (0.19.3)  
**Priority**: P1  

---

## Main goals

- After a stop, the next utterance fires `on_audio_start` again, so a host waiting for speech to begin sees it begin.

---

## Context / problem

Found by the AbstractAssistant desktop worker (release-closing wave 2026-10-28): the desktop showed its "thinking"
spinner while the voice was already speaking. `NonBlockingAudioPlayer._audio_callback`
(`abstractvoice/tts/tts_engine.py`) sets `_audio_started = True` when the first samples of an utterance are written
and fires `on_audio_start` only on the False -> True transition. The flag was cleared when playback drained and when
a stale stream was reopened, and by `AdapterTTSEngine.stop(close_stream=False)`, but never by
`stop_stream()` (the default stop path, `close_stream=True`, and the device-change restart). After a stop mid-utterance
the flag stayed True, so the next utterance played without `on_audio_start` (`vm.on_audio_start`).

## What changed

- `NonBlockingAudioPlayer.stop_stream()` sets `_audio_started = False` with the other playback state
  (`is_playing`, pause, queue).
- `docs/api.md` (Callbacks & hooks) and `llms-full.txt` state that `on_audio_start` fires once per utterance, the first
  one after a stop included.

## Validation

- `tests/test_audio_player_continuity.py::test_stop_stream_resets_started_so_the_next_utterance_fires_start` drives the
  real output callback with synthetic PCM (no device, no weights): start fires, `stop_stream()`, a new chunk fires
  start again. RED on the 0.19.2 code (`events == ["start"]`, the second start missing), GREEN with the fix.
- Local CI fast suite (`pytest -m "not integration and not model_download"`, Python 3.12): 881 passed, 3 skipped
  (6 `test_tts_reply_language` failures on the first run were the stale local venv missing the declared
  `py3langid` dependency; green once installed).
- Adversarial review (opus, read-only): ship. The desktop calls `player.stop_stream()` directly
  (AbstractAssistant `core/gateway_voice_manager.py`), the path fixed here; `VoiceManager.stop_speaking`
  (`close_stream=False`) already reset the flag. Device-change restart: no spurious extra start.

## Follow-ups (pre-existing, not changed here)

- `on_audio_end` never fires on a stop (both stop paths set `is_playing = False` first, and the end branch needs it
  True): a host tracking "speaking" only through start/end stays "speaking" after a stop.
- `stop(close_stream=False)` keeps the stream running: a callback already past the write can set the flag back to
  True after the reset (narrow window; closing it means handling the flag under a lock).
