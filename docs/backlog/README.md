## Backlog (how we work)

This folder is the **single source of truth** for planned and completed work in AbstractVoice.

## Current snapshot

- Updated: 2026-05-20
- Planned: 16
- Proposed: 9
- Completed: 28
- Recurrent: 1

### Principles

- **Backlog items are self-contained**: each task includes the outcome of research, key constraints, design choices, and the reasons for those choices (with references).
- **No test-driven special casing**: implementation must be general-purpose logic, not tailored to tests.
- **Keep the public contract stable**: preserve the integrator-facing API (`VoiceManager.speak/listen/transcribe/pause/resume/stop/...`).
- **Prefer permissive licensing**: only adopt MIT/Apache/BSD-compatible components. If none exist, document feasibility and create a backlog item.

---

## Folder layout

- `docs/backlog/planned/`: work to do next
- `docs/backlog/proposed/`: plausible ideas and follow-ups not yet committed
- `docs/backlog/completed/`: finished tasks with a completion report
- `docs/backlog/recurrent/`: checklists to run after development cycles

For larger work tracks, a topical subfolder under `planned/` or `proposed/` is
allowed when it improves readability and the folder includes a short
`README.md` that explains scope, lifecycle state, and recommended sequencing.

## Current planned Scenema track

These items were promoted from `docs/backlog/proposed/scenema/` on 2026-05-20.
Only the runtime spike is Scenema-specific; the other items are broader
prerequisites that remain reusable beyond Scenema. The internal
request/capability seam and shared torch runtime seam already exist on the
`scenema` branch, so the remaining planned work is now a rollout/hardening
track rather than a greenfield design track. Recommended implementation order:

1. `docs/backlog/planned/scenema/044_package_owned_directed_speech_request_and_capabilities.md`
2. `docs/backlog/planned/scenema/045_shared_runtime_foundation_for_advanced_speech_engines.md`
3. `docs/backlog/planned/scenema/046_scenema_class_python_runtime_spike.md`
4. `docs/backlog/planned/scenema/047_device_agnostic_runtime_and_apple_silicon_path.md`

Rationale:

- finish the package-owned request rollout before widening runtime and
  integration work further;
- harden the shared dependency/runtime foundation before adding another heavy
  optional backend;
- prove a Linux/CUDA rewrite path before claiming a serious Apple/MPS path;
- keep the former planning-boundary work folded into `044`, where ADR 0007 and
  the rollout now make it enforceable instead of duplicative.

## Current proposed DramaBox track

These items are grouped under `docs/backlog/proposed/dramabox/`. They are
still proposal-grade because the non-vendored LTX-family runtime spike remains
the main stop/go gate. Recommended reading order:

1. `docs/backlog/proposed/dramabox/043_dramabox_nonvendored_feasibility.md`
2. `docs/backlog/proposed/dramabox/049_package_owned_expressive_prompt_mapping_and_reference_audio_conditioning.md`
3. `docs/backlog/proposed/dramabox/050_advanced_speech_output_layout_and_result_metadata.md`
4. `docs/backlog/proposed/dramabox/051_ltx_audio_runtime_spike_for_dramabox.md`
5. `docs/backlog/proposed/dramabox/052_dramabox_clone_engine_integration.md`
6. `docs/backlog/proposed/dramabox/053_dramabox_base_tts_engine_and_capability_rollout.md`
7. `docs/backlog/proposed/dramabox/054_dramabox_apple_silicon_validation_and_explicit_degradation.md`

Rationale:

- capture reusable abstractions even if DramaBox never ships;
- keep the runtime spike separate from later engine-surface work;
- keep Apple validation explicitly last instead of silently assumed.

---

## Naming convention

Create new tasks as:

- `docs/backlog/planned/{NNN}_{short_description}.md`
- `docs/backlog/planned/{track}/{NNN}_{short_description}.md` for established
  multi-item tracks

Rules:
- **NNN**: zero-padded integer (e.g. `017`)
- **short_description**: snake_case, concise

Note: older tasks may use legacy names; do not rename unless a dedicated cleanup task exists.

---

## Task template (required sections)

Every backlog item must include:

- **Title**
- **Date**
- **Status**: Planned / Completed
- **Priority**: P0/P1/P2
- **Main goal(s)** and **secondary goal(s)**
- **Context / problem statement**
- **Constraints**
- **Research & references** (links + key findings)
- **Decision** (what we choose + why)
- **Dependencies** (ADRs and other tasks)
- **Implementation plan** (small steps, minimal surface area)
- **Success criteria**
- **Test plan**
- **Report** (for completed tasks only)

See `docs/backlog/template.md`.

---

## Work process (development cycle)

0. **Think, design, and plan** with long-term consequences in mind; prefer the cleanest, simplest, most efficient approach.
1. **Write the task** in `docs/backlog/planned/{NNN}_{short_description}.md`
   or `docs/backlog/planned/{track}/{NNN}_{short_description}.md`
   (self-contained, with research and decisions).
2. **Implement** the task.
3. **Test** and fix issues.
4. **Only when all tests pass**, move the task to `docs/backlog/completed/` and add the completion report at the end.
5. **Check recurrent tasks** (see `docs/backlog/recurrent/`) before claiming completion.
6. After completion, **bump semantic version** and add a **CHANGELOG** entry (this is a recurrent task).

Proposed items should be promoted before implementation when they become
accepted work. If an agent completes a small proposed item in one cycle, the
completed report must say it was promoted from `docs/backlog/proposed/`.
If a whole topical proposal track is accepted, move the existing files into
`docs/backlog/planned/{track}/` and update the track `README.md` instead of
duplicating them.
