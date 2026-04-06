# Development notes (internal)

This document is for contributors. User-facing docs live in:

- `README.md`
- `docs/repl_guide.md`
- `docs/installation.md`

## Layout

- `abstractvoice/vm/` — `VoiceManager` façade + mixins (TTS/STT/cloning orchestration)
- `abstractvoice/adapters/` — adapter implementations (Piper / AudioDiT / OmniVoice TTS; Faster-Whisper STT)
- `abstractvoice/audiodit/` — AudioDiT runtime + HF model implementation (vendored code; avoids `trust_remote_code`)
- `abstractvoice/omnivoice/` — OmniVoice runtime wrapper (offline-first + device/dtype policy glue)
- `abstractvoice/tts/` — audio playback utilities (`NonBlockingAudioPlayer`)
- `abstractvoice/cloning/` — optional cloning engines + voice store (`f5_tts` / `chroma` / `audiodit` / `omnivoice`)
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
- Torch engine snapshots are resolved offline-first in their runtimes (`abstractvoice/audiodit/runtime.py`, `abstractvoice/omnivoice/runtime.py`).
- Cloning downloads are explicit per engine (`abstractvoice/cloning/engine_f5.py`, `abstractvoice/cloning/engine_chroma.py`, `abstractvoice/cloning/engine_audiodit.py`, `abstractvoice/cloning/engine_omnivoice.py`).

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
python -m pytest -q
```
