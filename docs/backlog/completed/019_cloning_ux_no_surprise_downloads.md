## Task 019: Cloning UX hardening (no surprise downloads)

**Date**: 2026-01-23  
**Status**: Completed  
**Priority**: P0  

---

## Problem

The REPL could accidentally trigger multi‑GB downloads (OpenF5 artifacts) during normal chat if a cloned voice was selected, polluting output and feeling like a hang/crash.

Also, cloning store operations (list/clone metadata) were initializing heavy components (STT engine), creating additional surprise downloads.

---

## What changed

- **Cloning engine is now lazy**
  - `F5TTSVoiceCloningEngine` no longer instantiates faster‑whisper at import/init time.
  - `VoiceCloner` no longer instantiates the engine at init time; it creates it only when synthesis is requested.

- **Store can keep `reference_text`**
  - `VoiceManager.clone_voice(..., reference_text=...)` is supported.
  - REPL `/clone-my-voice` stores the prompt text so we don’t need STT for that flow.

- **REPL UX guardrails**
  - Normal chat will not trigger cloning downloads:
    - If a cloned voice is selected but cloning runtime is not ready, REPL prints a short hint and skips TTS for that response.
  - Added explicit commands:
    - `/cloning_status`
    - `/cloning_download` (explicitly downloads OpenF5 artifacts; expected to be noisy/slow)
  - Removed auto-selection of `hal9000` on startup (avoids surprise downloads). The voice is seeded in the store, but users must explicitly select it.
  - `/clone ...` no longer auto-selects the cloned voice. It instructs users to select it after runtime setup.

---

## How to use (REPL)

- Check cloning readiness:
  - `/cloning_status`
- Download the cloning model once:
  - `/cloning_download`
- Select and use a cloned voice:
  - `/clones`
  - `/tts_voice clone hal9000`
  - `/speak Good evening, Dave.`

---

## Validation

- Tests: **30 passed, 2 skipped**

