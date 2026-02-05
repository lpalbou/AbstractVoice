## Task 032: Markdown speech sanitization (natural TTS in REPL)

**Date**: 2026-02-05  
**Status**: Completed  
**Priority**: P1  

---

## Main goals

- Make spoken output sound natural when the assistant returns Markdown-formatted text.

Scope (intentionally minimal):
- Strip ATX headers (`#`…`#####`) at the start of a line.
- Strip emphasis markers (`**bold**`, `*italic*`).

---

## Context / problem

LLM responses often include Markdown (headers/emphasis). When spoken verbatim, TTS engines can pronounce formatting tokens (e.g. “hash”, “asterisk”), which sounds unnatural.

In the REPL we speak raw assistant text:

- `abstractvoice/examples/cli_repl.py` → `_speak_with_spinner_until_audio_starts()` → `VoiceManager.speak(...)`

---

## Constraints

- **No breaking changes** to the supported API surface (`docs/api.md`): `VoiceManager.speak()` must speak exactly what it’s given.
- Prefer **REPL-only** behavior by default (the primary “assistant speaks LLM output” path).
- Keep it lightweight (regex only; no Markdown parser dependency).

---

## Decision

**Chosen approach**: regex-based sanitizer, applied only in the REPL right before calling `VoiceManager.speak()`.

**Why**:
- Fixes the UX where it happens (assistant output spoken in REPL) without changing library semantics for integrators.
- Minimal implementation surface and easy to test.

---

## Implementation

- Added `abstractvoice/text_sanitize.py`:
  - `sanitize_markdown_for_speech(text: str) -> str`
  - Removes:
    - headers at line start (`#`…`#####`) while preserving the header text
    - emphasis markers: `**...**` and `*...*`
- Wired into REPL speech:
  - `abstractvoice/examples/cli_repl.py`: sanitize `text` in `_speak_with_spinner_until_audio_starts()` before calling `self.voice_manager.speak(...)`.
- Added unit tests:
  - `tests/test_text_sanitize_markdown.py`

---

## Success criteria (met)

- REPL TTS no longer pronounces `#`, `**`, `*` for common Markdown output.
- No change to `VoiceManager.speak()` for non-REPL integrators.
- Unit tests cover the sanitizer and pass.

---

## Test plan

- `python -m pytest -q`

---

## Report

### Summary

Implemented a minimal Markdown sanitizer for spoken output in the REPL (headers + emphasis only) to keep TTS natural while preserving printed output unchanged.

### Validation

- Tests: `python -m pytest -q` (pass)

