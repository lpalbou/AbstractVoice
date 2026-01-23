## Task 016: ADR 0002 Phase 2 (optional AEC-based barge-in)

**Status**: Completed  
**Priority**: P1  

---

## What was implemented

### Optional dependency extra

- Added `abstractvoice[aec]` extra in `pyproject.toml` (uses `aec-audio-processing`).

### AEC module (opt-in)

- Added `abstractvoice/aec/` with `WebRtcAecProcessor` wrapper:
  - Works on PCM16 bytes and follows the “reverse stream first” pattern required by WebRTC APM.
  - Keeps imports lazy so normal installs remain clean.

### Full-duplex pipeline wiring

- Added a **speaker output tap** in `NonBlockingAudioPlayer` (`on_audio_chunk`) so we can capture the actual audio being written to the output stream.
- `VoiceManagerCore` forwards those chunks to the active `VoiceRecognizer` (if present) via `feed_far_end_audio(...)`.
- `VoiceRecognizer` (when AEC enabled) applies echo cancellation on mic input before VAD/STT:
  - far-end audio is resampled to mic sample rate if needed
  - mic chunks are split into 10ms frames for APM robustness

### Barge-in semantics

- In `voice_mode="full"`:
  - If AEC is enabled (`voice_recognizer.aec_enabled=True`), we **do not pause** `tts_interrupt` during TTS playback.
  - If AEC is not enabled, we keep the previous safe behavior (pause interrupt to avoid self-interruption).

### Public API

- Added `VoiceManager.enable_aec(enabled=True, stream_delay_ms=0)` to opt in.

---

## How to use

1) Install optional AEC:

```bash
pip install "abstractvoice[aec]"
```

2) Enable AEC + full mode barge-in:

```python
from abstractvoice import VoiceManager

vm = VoiceManager()
vm.enable_aec(True, stream_delay_ms=0)
vm.set_voice_mode("full")
vm.listen(on_transcription=lambda t: print("USER:", t))
```

Now, when the user starts speaking during TTS playback, `tts_interrupt_callback` can fire without self-interruption (best-effort; depends on device/audio routing).

---

## Validation

- Tests: **32 passed, 2 skipped**

