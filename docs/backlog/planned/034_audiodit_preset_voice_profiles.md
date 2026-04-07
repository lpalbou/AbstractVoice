## Task 034: AudioDiT preset voice profiles (male/female sets)

**Date**: 2026-04-06  
**Status**: Planned  
**Priority**: P2  

---

## Main goals

- Provide a small set of **repeatable “voice profiles”** for AudioDiT TTS:
  - **3 male** presets
  - **3 female** presets
- Make the profiles **easy to select** (REPL + library usage).

## Secondary goals

- Make profile behavior predictable across turns (stable “speaker identity” within a discussion).
- Document limitations clearly (AudioDiT upstream is not primarily a voice-design model).

---

## Context / problem

AudioDiT in AbstractVoice is used as:
- a base TTS engine (`tts_engine="audiodit"`)
- and a cloning backend (prompt audio + transcript)

Unlike OmniVoice, AudioDiT does **not** expose a first-class “voice design prompt” surface (gender/age/etc.) in the current integration.
What we do have:
- `seed` (best-effort determinism)
- a **session prompt** mechanism (prompt-audio + prompt-text prefix) to keep a stable speaker identity across turns

Users still want a quick way to get a few distinct “default voices” without doing explicit cloning.

---

## Constraints

- Do not ship model weights or large audio assets in-repo.
- AudioDiT’s perceived voice identity may shift with:
  - accelerator nondeterminism (MPS/CUDA)
  - model revisions
  - text content (prosody/pitch drift)
- The “male/female” label is likely **subjective** unless upstream provides a grounded control.

---

## Proposed approaches (choose one)

### Option A (recommended): “seeded anchor” profiles (no extra assets)

Define each profile as:
- a fixed `seed`
- a fixed **anchor phrase** (short, stable sentence)
- a fixed `session_prompt_seconds`
- a quality preset (`fast|balanced|high`)

Mechanism:
- When a profile is selected, we synthesize the anchor phrase once and use it to seed the session prompt cache.
- Subsequent `/speak` calls use the session prompt to maintain a stable speaker identity.

Pros:
- No extra downloads beyond AudioDiT itself.
- Fast setup; works with current integration.

Cons:
- “male/female” is not guaranteed; might require curation by listening tests.

### Option B: ship tiny generated anchor WAVs (not recommended by default)

Generate and store short anchor WAVs (6 files) and use them as prompt audio.

Pros:
- More consistent anchors (no first-use anchor generation variance).

Cons:
- Adds binary assets to the repo and introduces licensing/provenance questions.

---

## Proposed design

### Dependency: shared “voice profile” abstraction (Task 036)

AudioDiT profiles should be exposed through the same cross-engine interface as other engines (list/apply/show) so that:
- the REPL can expose a single `/profile ...` surface (no `/audiodit profile ...` special-casing)
- AbstractCore and third-party integrations can forward `profile_id` without engine-specific glue

See: `docs/backlog/planned/036_voice_profile_abstraction.md`.

Minimum viable profile payload:
- `engine: "audiodit"`
- `name: str`
- `params`:
  - `seed: int`
  - `quality_preset: "fast"|"balanced"|"high"`
  - `anchor_text: str`
  - `session_prompt_seconds: float`

Profiles should be stored as a JSON asset and applied through:
- `AudioDiTTTSAdapter` settings (`AudioDiTSettings.seed`)
- session prompt initialization (anchor synthesis)

---

## Implementation plan

- Add a built-in profile list (JSON asset + loader).
- Add an adapter-level method to “apply profile”:
  - set seed
  - reset any session prompt state
  - synthesize anchor text once (no playback), capture session prompt
- Add REPL UX:
  - `/profile list` (when `tts_engine=audiodit`)
  - `/profile <name>` (applies to the active `tts_engine`)
- Document caveats:
  - French is not guaranteed
  - “male/female” labeling is best-effort and may need revision

---

## Success criteria

- A user can select a profile and get a stable-sounding AudioDiT voice across turns without needing cloning.
- The implementation does not add large assets and does not create surprise downloads.

