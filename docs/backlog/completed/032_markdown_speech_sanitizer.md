## Task 032: Markdown speech sanitization (natural TTS in REPL)

**Date**: 2026-02-05  
**Status**: Completed  
**Priority**: P1  

---

## Update (2026-02-06)

Markdown sanitization is now applied by default for *all* integrators via:

- `VoiceManager.speak(..., sanitize_syntax=True)` (default)
- `VoiceManager.speak_to_bytes(..., sanitize_syntax=True)` (default)
- `VoiceManager.speak_to_file(..., sanitize_syntax=True)` (default)

To speak raw text (including Markdown tokens), pass `sanitize_syntax=False`.

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

- Prefer behavior that keeps spoken output natural by default (integrators can opt out per call).
- Keep it lightweight (regex only; no Markdown parser dependency).

---

## Decision

**Chosen approach**: regex-based sanitizer, applied by default in `VoiceManager` TTS entrypoints (opt-out per call).

**Why**:
- Keeps spoken output natural across REPL + headless/server usage.
- Minimal implementation surface and easy to test.

---

## Implementation

- Added `abstractvoice/text_sanitize.py`:
  - `sanitize_markdown_for_speech(text: str) -> str`
  - Removes:
    - headers at line start (`#`…`#####`) while preserving the header text
    - emphasis markers: `**...**` and `*...*`
- Wired into `VoiceManager` TTS entrypoints (default behavior):
  - `abstractvoice/vm/tts_mixin.py`: `speak()`, `speak_to_bytes()`, `speak_to_file()` accept `sanitize_syntax: bool = True`
- Added unit tests:
  - `tests/test_text_sanitize_markdown.py`
  - `tests/test_tts_sanitize_syntax_default.py`

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
