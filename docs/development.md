# Development notes (internal)

This document is for contributors. User-facing docs live in:

- `README.md`
- `docs/repl_guide.md`
- `docs/installation.md`

## Layout

- `abstractvoice/vm/` — `VoiceManager` façade + mixins (TTS/STT/cloning orchestration)
- `abstractvoice/adapters/` — adapter implementations (Piper TTS, Faster-Whisper STT)
- `abstractvoice/tts/` — audio playback utilities (`NonBlockingAudioPlayer`)
- `abstractvoice/cloning/` — optional cloning engines + voice store (F5 / Chroma)
- `abstractvoice/examples/` — REPL and demo entrypoints

## Offline-first policy

- The REPL (`python -m abstractvoice cli`) runs with `allow_downloads=False`.
- Any network download must be explicit:
  - `python -m abstractvoice download ...`
  - `abstractvoice-prefetch ...`
  - REPL: `/cloning_download ...`

Implementation points:

- Piper downloads are gated in `abstractvoice/adapters/tts_piper.py`.
- Faster-Whisper offline mode is enforced in `abstractvoice/adapters/stt_faster_whisper.py`.
- Cloning downloads are explicit per engine (`abstractvoice/cloning/engine_f5.py`, `abstractvoice/cloning/engine_chroma.py`).

## Audio playback + prompt hygiene

`abstractvoice/tts/tts_engine.py` provides:

- `NonBlockingAudioPlayer` (pause/resume/stop)
- `_SilenceStderrFD` to suppress OS-level stderr spam that can corrupt terminal UI

The REPL avoids printing the prompt manually to prevent duplicate prompts (`> >`).

## Cloned TTS (streaming + cancellation)

Cloned synthesis runs in a background thread in `abstractvoice/vm/tts_mixin.py`:

- cancellation token per utterance (`_cloned_cancel_event`)
- optional streaming (`cloned_tts_streaming`)
- per-utterance metrics are recorded for verbose REPL output

## Memory management (important)

Cloning engines can be very large (especially Chroma). The REPL:

- unloads other cloning engines when switching cloned voices
- unloads the Piper voice while using cloned voices

Core support:

- `abstractvoice/cloning/manager.py`: engine cache + `unload_*` helpers
- engines implement `unload()` best-effort (GC + torch cache clears)

## Tests

```bash
pytest -q
```

