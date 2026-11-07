# 0067-abstractvoice: [IMPROVEMENT] Make clone validation and segment retries explicit
> Deprecated: 2026-11-07 — superseded by the operator's ruling and its implementation. Ruling (2026-11-05/07): "retry once,
> and if it failed again, still use that voice because a small delta is better than to default to a random voice";
> "a hard failure is worse than playing the voice with a small delta". Implemented on main (14da7d3, d0abd24): an opening
> under the bar is regenerated once and then spoken anyway; later windows are scored for the record only (`low_windows`);
> the engine never raises `ClonedVoiceNotAppliedError`; the check and its bar are constructor/plugin settings
> (`qwen3_tts_speaker_check`, `qwen3_tts_min_speaker_similarity`), never environment variables. No request-scoped
> retry policy is wanted ("do not over complexify"). Calibration of short-utterance scores remains a research note only.

> Created: 2026-11-06
> Type: improvement
> Priority: P2
> Labels: voice, cloning, latency

**Date**: 2026-11-06  
**Status**: Deprecated (2026-11-07)  
**Priority**: P2

- Work-item id: abstractvoice-0067
- Owner: framework (intake; implementation belongs to AbstractVoice)
- Thread: Unavailable; agora connection refused at intake.
- Hub refs: None; mirror registration pending hub availability.

## Summary

Expose a package-owned, request-scoped policy for clone similarity checking and
same-clone segment regeneration. Let callers explicitly choose retry behavior
instead of incurring hidden regeneration or substituting a default voice.
Calibrate marginal and short-utterance scores before choosing policy defaults.

## Main goals

- Make validation and retry decisions separately configurable and discoverable.
- Retry only the failed, not-yet-delivered segment with the original clone when
  regeneration is explicitly enabled; bound attempts and cancellation latency.

## Secondary goals

- Distinguish a failed similarity check from proven perceptual voice mismatch.
- Report attempt count, generation latency, scores and the effective policy.

## Context / problem

On 2026-11-05, a short Assistant reply beginning “Hey! Yeah, I'm here and working”
scored 0.945 and 0.941 against an opening threshold of 0.950 using Qwen3-TTS
0.6B Base. Both attempts were rejected. The app then dropped voice pins and
generated a default voice, adding latency and losing the requested identity.
Assistant commit `270f37f` removed that app fallback: selected-voice failures now
wait for explicit Play. It does not change the engine's two attempts.
The scores alone do not establish that the generated voice sounded wrong.

## Current code reality

- `abstractvoice/cloning/engine_qwen3_tts.py`: `SPEAKER_RETRIES = 1` means two
  attempts; `_checked_generation` already regenerates the same clone segment on an
  opening rejection. Do not implement that behavior a second time.
- Engine construction already honors `ABSTRACTVOICE_QWEN3_TTS_SPEAKER_CHECK` and
  `ABSTRACTVOICE_QWEN3_TTS_MIN_SPEAKER_SIMILARITY`. Missing: a supported,
  discoverable request-scoped check/retry policy across public synthesis paths.
- `ClonedVoiceNotAppliedError` carries segment index/count, scores, threshold,
  mid-segment state and segment records. Its prose currently claims another
  voice from the score; new diagnostics should distinguish measurement/inference.
- `abstractvoice/integrations/abstractcore_plugin.py::tts_stream` emits segment
  offsets and `segment_end` (completed item 0066, ADR 0009). Boundary resume
  exists. A failed opening may have delivered no audio; a mid-segment failure
  can have delivered audio and must not be treated as an untouched segment.

## Constraints

- No implementation is authorized by this proposal; the operator deferred work
  at Gateway level and below on 2026-11-06.
- Keep voice/profile/provider/model identity fixed through retries. No automatic
  stock-voice substitution; no whole-reply regeneration to retry one segment.
- Keep settings per request; shared cached engines must not leak policy between users.
- Do not globally lower 0.950 or disable checks based on this single incident.

## Research, options, and references

- **Explicit retry budget**: zero automatic retries versus a bounded number of
  same-clone retries; both expose attempts and time spent. Prefer a clear budget
  over coupling retry to a general quality preset.
- **Validation modes to evaluate**: enforce, report-only, and disabled. Report-only
  could deliver the original generated audio with uncertainty metadata, avoiding
  needless recomputation; it needs explicit caller choice and clear semantics.
- Keep similarity policy distinct from silence, invalid audio and runaway/repetition
  protection. Investigate short clips, languages, reference quality and backend
  calibration using real weights; do not infer a mismatch probability from cosine scores.
- Read completed `056_cloned_voice_checked_per_generation.md`,
  `057_clone_speaker_check_streams_continuously.md`, and
  `0066_tts_stream_text_segments_for_exact_resume.md` before changing behavior.

## Decision

Proposed, not committed. Agree defaults and compatibility behavior before promotion.
The operator's desired direction is optional regeneration of the failed segment
with the same clone. Whether report-only delivery is desirable needs evaluation.

## Dependencies and ADR status

- ADRs 0004 (streaming/cancellation), 0006 (package-owned selectors), 0007
  (package-owned requests), 0008 (Qwen backend), 0009 (segment resume).
- ADR impact: document explicit validation/retry policy before implementation
  closure; revise ADR 0009 only if its delivery/resume semantics actually change.
- Gateway follow-up: [abstractgateway-1010](../../../../abstractgateway/docs/backlog/proposed/1010_typed_clone_failures_and_explicit_segment_retry.md).
  Public forwarding needs Core/Runtime integration;
  Gateway must not import the engine or own its thresholds.

## Implementation plan

1. Reproduce/calibrate borderline and clearly wrong-voice cases with real weights.
2. Specify request policy, defaults, capabilities and typed failure/outcome metadata.
3. Implement opt-in bounded segment retry and policy isolation in package-owned paths.
4. Integrate through Core/Runtime; document adoption and compatibility.

## Acceptance Criteria

- [ ] Disabled automatic retry generates once; enabled retry regenerates only the
  failed segment with the same clone and respects its documented budget.
- [ ] Accepted prior segments are not regenerated; cancellation prevents new attempts;
  mid-segment delivery has an explicit, tested replay policy.
- [ ] Report-only/disabled checking, if adopted, is explicit and returns honest
  validation metadata; invalid-audio protections remain separately defined.
- [ ] Concurrent requests with different policies do not affect each other.
- [ ] Real-weight evidence records false rejection/mismatch cases, attempts,
  first-audio latency and total latency across representative clones and short replies.

## Testing

- `python -m pytest tests/test_qwen3_tts_clone_speaker_check.py tests/test_qwen3_tts_clone_stream_continuity.py tests/test_tts_stream_segments.py -q`
- Add policy/default/cancellation/concurrency regression coverage. Run an explicitly
  configured real-weight Qwen harness; mocked scores alone cannot justify new defaults.

## Promotion criteria

Operator schedules lower-level work, package owners agree the policy/defaults,
and calibration plus the integration boundary are specified. Expected outcome:
callers control extra computation without silently losing voice identity.

## Report

Not implemented. Intake records current code and desired follow-up only.
