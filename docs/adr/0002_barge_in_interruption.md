## ADR 0002: Barge-in (Interrupt TTS When User Starts Speaking)

**Date**: 2026-01-23  
**Status**: Accepted (phased)

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

We support “interactive discussion” with **two modes** that are robust without AEC, and a **stop phrase**:

- **wait mode (recommended default)**: pause listening during TTS playback (no self-interruption).
- **full mode**: keep listening during TTS but **disable auto-interrupt** to avoid self-interruption; user can still stop via explicit commands (REPL `/stop`) or UI controls.
- **Stop phrase interruption (enabled)**: while listening, saying **“ok stop”** (and compatible variants like “okay stop”) stops the assistant safely without requiring echo cancellation.

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

- default to **wait mode** (no self-interruption)
- allow instant user control via:
  - REPL `/stop`, `/pause`, `/resume`
  - optional keyboard push-to-talk / push-to-stop

And (only if we can do it robustly) add an “echo-aware VAD” heuristic in the future
(correlation against recently played audio) to approximate barge-in without heavy deps.

