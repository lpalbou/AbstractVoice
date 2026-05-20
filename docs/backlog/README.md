## Backlog (how we work)

This folder is the **single source of truth** for planned and completed work in AbstractVoice.

## Current snapshot

- Updated: 2026-05-20
- Planned: 12
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

For larger proposal tracks, a topical subfolder under `proposed/` is allowed
when it improves readability and the folder includes a short `README.md` that
explains scope and promotion order.

## Current proposed Scenema track

These items form the current proposal set created from the Scenema evaluation.
Only the runtime spike is Scenema-specific; the other items are broader
prerequisites that remain reusable beyond Scenema. Recommended promotion order:

1. `docs/backlog/proposed/scenema/044_package_owned_directed_speech_request_and_capabilities.md`
2. `docs/backlog/proposed/scenema/048_abstractvoice_owned_speech_planning_boundary.md`
3. `docs/backlog/proposed/scenema/045_shared_runtime_foundation_for_advanced_speech_engines.md`
4. `docs/backlog/proposed/scenema/046_scenema_class_python_runtime_spike.md`
5. `docs/backlog/proposed/scenema/047_device_agnostic_runtime_and_apple_silicon_path.md`

Rationale:

- define package-owned request semantics before integrating a scene-aware engine;
- keep `abstractvoice` as the owner of speech planning semantics instead of
  pushing them upward into `abstractcore`;
- agree on the shared dependency/runtime foundation before adding another heavy
  optional backend;
- prove a Linux/CUDA rewrite path before attempting Apple/MPS portability.

---

## Naming convention

Create new tasks as:

- `docs/backlog/planned/{NNN}_{short_description}.md`

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
1. **Write the task** in `docs/backlog/planned/{NNN}_{short_description}.md` (self-contained, with research and decisions).
2. **Implement** the task.
3. **Test** and fix issues.
4. **Only when all tests pass**, move the task to `docs/backlog/completed/` and add the completion report at the end.
5. **Check recurrent tasks** (see `docs/backlog/recurrent/`) before claiming completion.
6. After completion, **bump semantic version** and add a **CHANGELOG** entry (this is a recurrent task).

Proposed items should be promoted before implementation when they become
accepted work. If an agent completes a small proposed item in one cycle, the
completed report must say it was promoted from `docs/backlog/proposed/`.
