## ADR 0002: Barge-in (Interrupt TTS When User Starts Speaking)

**Date**: 2026-01-23  
**Status**: Accepted (phased)

**Amendment (2026-02-04)**: Phase 1 shipped with an explicit `stop` mode (stop-phrase barge-in without AEC) and a stricter `wait` mode (pause mic processing during TTS). The sections below reflect current behavior and implementation.

---

## Context / Problem

We want a natural conversation: while the assistant is speaking, the user can start speaking and the assistant stops immediately (“barge-in”).

The hard part is that the microphone hears:
- user speech (desired)
- the assistant audio coming from the speakers (echo/feedback)

Without **echo cancellation**, VAD/STT will often detect the assistant’s own voice and self-interrupt.

---

## Investigation (Jan 2026)

There are open-source acoustic echo cancellation (AEC) options (WebRTC APM, SpeexDSP), and Python bindings exist, but:

- Many require compilation toolchains, have limited platform wheels, or introduce install friction.
- Some are Linux/macOS only in practice (wheel availability varies).

This conflicts with our core constraint: **easy pip install on Windows/macOS/Linux**.

---

## Decision

### Phase 1 (default, out-of-box)

We support “interactive discussion” with **voice modes** that are robust without AEC, plus a **stop phrase**:

- **wait mode (turn-taking)**: pause mic processing during TTS playback. This is the most robust way to avoid self-interruption on speakers, but it means there is **no voice barge-in** (and no stop-phrase detection) while audio is playing.
- **stop mode (recommended on speakers)**: keep mic processing running, but during TTS:
  - suppress normal transcriptions
  - disable “interrupt on any speech”
  - keep a rolling stop-phrase detector active so the user can still say **“ok stop”** / **“okay stop”** to cut playback
- **full mode (barge-in by speech)**: keep listening and allow barge-in by detected speech. Best with AEC or a headset; on speakers, echo can still cause false barge-in (mitigated by heuristics).
- **Stop phrase interruption (available while listening)**: saying **“ok stop”** / **“okay stop”** stops TTS playback without requiring echo cancellation. A conservative bare **“stop”** is supported but requires confirmation to reduce false hits.

Rationale: without AEC, “interrupt on any speech” is unreliable because the mic hears the assistant audio too. A stop phrase is robust and works everywhere.

### Phase 2 (optional advanced barge-in)

Add optional barge-in via AEC as an **extra** (opt-in dependency group), if and only if:
- it is MIT/Apache/BSD licensed
- it has reliable wheels for the major platforms, or we accept it as “advanced”

Candidate packages to re-evaluate:
- WebRTC audio processing AEC bindings (PyPI packages emerging in 2025)
- SpeexDSP AEC bindings

---

## Best simulation when AEC is not available

If we cannot reliably deploy AEC out-of-box, the best “natural” simulation is:

- default to a **non-self-interrupting mode** (wait/stop, depending on UX needs)
- allow instant user control via:
  - REPL `/stop`, `/pause`, `/resume`
  - optional keyboard push-to-talk / push-to-stop

And (only if we can do it robustly) add an “echo-aware VAD” heuristic in the future
(correlation against recently played audio) to approximate barge-in without heavy deps.
