# ADR 0005: Torch device and dtype defaults come from shared compute helpers

Status: Accepted.

## Context

AbstractVoice now ships multiple torch-backed engines across TTS, STT, and
cloning. If each engine chooses its own default device and dtype independently,
the package becomes harder to reason about and platform bugs become harder to
debug.

The repo already has shared device and dtype helpers. The missing policy
question is whether engines should treat those helpers as advisory or
authoritative.

## Decision

- Automatic torch device selection starts from `abstractvoice/compute/device.py`
  and its `best_torch_device()` helper.
- Automatic torch dtype selection starts from
  `abstractvoice/compute/dtype.py` and its `best_torch_dtype_name()` /
  `resolve_torch_dtype()` helpers.
- Environment overrides remain first-class:
  - `ABSTRACTVOICE_TORCH_DEVICE`
  - `ABSTRACTVOICE_TORCH_DTYPE`
- Engine-specific exceptions are allowed only when the upstream runtime or model
  has a documented incompatibility. Those exceptions must narrow or override the
  shared default deliberately; they must not replace the shared policy with
  unrelated ad hoc heuristics.
- There is no repo-wide promise that every torch engine will transparently retry
  on CPU. If an engine cannot safely run on the chosen device/dtype, it must
  either:
  - fail clearly; or
  - implement and document an engine-local fallback.
- Device and dtype selection stay independent from download and offline-first
  policy.

## Consequences

### Positive

- New torch engines have one default policy to plug into.
- Debugging becomes easier because requested and resolved device/dtype behavior
  can be compared across engines.
- Power users keep explicit override knobs.

### Negative

- Some models still need engine-local caveats, especially on MPS.
- Shared defaults cannot capture every model-specific performance tradeoff.

### Neutral

- The policy sets the starting point for engines; it does not eliminate
  engine-specific runtime knowledge.

## Enforcement

- New torch-backed engines should call the shared compute helpers before adding
  special-case logic.
- PRs that add inline device/dtype policy without a model-specific justification
  should be treated as drift.
- Runtime info and bug reports should expose requested and resolved
  device/dtype where practical.

## Validation

- `tests/test_compute_dtype_policy.py`
- `tests/test_audiodit_smoke_optional.py`
- `tests/test_omnivoice_smoke_optional.py`

## Backlog links

- `docs/backlog/completed/030_chroma_4b_optional_s2s_and_cloning.md`

## Related

- `abstractvoice/compute/device.py`
- `abstractvoice/compute/dtype.py`
- [../benchmark.md](../benchmark.md)
