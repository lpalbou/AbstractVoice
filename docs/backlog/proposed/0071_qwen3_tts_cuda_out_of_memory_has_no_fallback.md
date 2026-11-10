## Task 0071: Qwen3-TTS raises on a full GPU where faster-whisper falls back and says why

**Date**: 2026-11-10  
**Status**: Proposed  
**Priority**: P2

---

## Main goal

One behaviour for every local engine when the GPU is present but has no free memory.

## Context / problem statement

Observed 2026-11-10 on the Linux test server while another process held 13.8 of the card's 15 GB:
faster-whisper (`adapters/stt_faster_whisper.py`) caught the CUDA failure, loaded on the
processor and reported `execution_device()` with the reason ("loading on CUDA failed (CUDA failed
with error out of memory); running on the processor"). Qwen3-TTS (`qwen3_tts/runtime.py`,
`resolve_torch_runtime(..., allow_cpu_fallback=self._device_pref == "auto")`) resolved `cuda`,
then `torch.OutOfMemoryError` propagated from the weight transfer: no fallback, no reason recorded.
Transformers ASR has its own fallback. Three engines, three behaviours.

## Decision

Proposed: when the device was `auto`, a CUDA out-of-memory during load falls back to the processor
with the reason recorded in `runtime_info()` (as faster-whisper does); an explicit `device="cuda"`
keeps raising. Whether a 0.6B model at float32 on a 4-core processor (rtf ~4) is a useful fallback
or should rather be a clear refusal is part of the decision: the operator's rule is loud over
silent, so the reason must reach the caller either way.

## Dependencies

- 0070 (decode loop), the dtype policy fix.

## Success criteria

- The three local torch engines report where they run and why through the same shape of
  `runtime_info()` / `execution_device()`; a unit test per engine with the CUDA load monkeypatched
  to raise `torch.OutOfMemoryError`.
