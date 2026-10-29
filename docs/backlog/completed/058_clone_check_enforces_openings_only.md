## Task 058: the clone check enforces openings only; windows after the opening are logged

**Date**: 2026-10-29  
**Status**: Completed (0.19.6)  
**Priority**: P1  

## Context / problem

0.19.5 stopped a segment when any 4 s window after its opening scored < 0.93 against the reference. On the operator's
gateway (merge desk, ~125 s text = one sentence pair x 9, hal9000-qwen2, idle GPU) 3 of 4 replies stopped part-way
(after 79, 62, 47 s of audio); 0.19.4 completed the same request. Lead ruling: enforce openings only, log the windows.

## What changed

- After a verified opening, chunks are released as generated through the reply's `_Outbox` (0.19.5 continuous
  delivery kept). Each 4 s window is scored into the segment record (`window_scores`, `low_windows`) and logged at INFO
  when below 0.93; nothing is stopped or regenerated on it.
- Openings: unchanged (held for the reply's first segment; later openings checked while the previous plays; one
  regeneration, then `ClonedVoiceNotAppliedError`). A generation shorter than the opening window is decided at its end.

## Validation (isolated process, real weights, 0.6B Base MLX 8-bit, the merge desk's text; NOT the operator's gateway)

| code | voice | run | result | audio | first audio | gaps | logged low windows |
|---|---|---|---|---|---|---|---|
| 0.19.5 | hal9000-qwen2 | 0 | STOP, segment 2 tail (windows 0.933, 0.886) | 97.0 s | 0.78 s | 0 | - |
| 0.19.5 | hal9000-qwen2 | 1-3 | completed | 107-117 s | 0.64-1.06 s | 0 | - |
| 0.19.6 | hal9000-qwen2 | 0 | completed | 117.2 s | 0.86 s | 0 | 1 (0.897, isolated) |
| 0.19.6 | hal9000-qwen2 | 1 | completed | 109.4 s | 1.09 s | 0 | 0 |
| 0.19.6 | hal9000-qwen2 | 2 | STOP at segment 1 opening (0.936, then 0.945 < 0.95) | 0 s | - | - | - |
| 0.19.6 | hal9000-qwen2 | 3 | "completed" with a RUNAWAY: 234 s (segment 2 missed EOS) | 234.2 s | 3.24 s | 0 | 24 (0.85-0.89) |
| 0.19.6 | sisko | 0-3 | completed | 96-100 s | 0.61-2.47 s | 0 | 0 (min 0.932) |

This does NOT meet the acceptance bar (4/4 completed) on hal9000-qwen2: run 2 stopped at the opening (the 0.19.4 rule;
hal's genuine openings sit near 0.95), and run 3 played ~124 s of a runaway in another voice (offline 4 s windows
0.85-0.89 against the reference, 0.88-0.91 against the reply's own audio) and reported success. Mid-segment drift does
occur without an opening drift (runaway generations).

Tests: `test_a_voice_change_part_way_is_recorded_not_stopped`, `test_a_mid_segment_dip_neither_stops_nor_gaps_the_reply`
(both red on 0.19.5); the opening, continuity and record tests stay green. Fast suite: 901 passed.
