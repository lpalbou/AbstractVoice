# ADR 0001: Local assistant installs must work out of the box

Status: Accepted. Amended 2026-05-08.

## Context

AbstractVoice supports two real deployment shapes:

- a lightweight remote/plugin install for OpenAI and OpenAI-compatible audio;
- an explicit local assistant install for microphone capture, STT, TTS, and
  interactive playback on a user machine.

The project originally treated `pip install abstractvoice` as the local
assistant install. That is no longer true. The base package is now
remote-first, while local voice runtimes live behind explicit install profiles
and granular extras.

That packaging change did not remove the local assistant goal. It narrowed the
scope of the goal: the out-of-box experience now applies to explicit local
installs such as `abstractvoice[apple]`, `abstractvoice[gpu]`, or equivalent
granular compositions, not to the lightweight base package.

## Decision

- The base package stays lightweight and remote-first.
- A first-class local assistant experience remains required through explicit
  local install profiles and compatible granular extras.
- Local assistant installs should provide mic capture, playback, VAD, local STT,
  and local TTS without custom source builds in the normal supported path when
  upstream wheels and standard OS audio packages are available.
- The preferred local assistant stack remains:
  - `sounddevice` / PortAudio for audio I/O;
  - `webrtcvad` for VAD;
  - `faster-whisper` for local STT;
  - explicit local TTS engines such as Piper or Supertonic.
- `VoiceManager()` and lightweight plugin/server usage stay remote-first by
  default. Local behavior must be selected explicitly through install extras and
  engine selection.

## Consequences

### Positive

- Remote/plugin deployments keep a small default dependency footprint.
- Local assistant installs remain a supported product shape instead of an
  accidental side effect of the base package.
- The repo can optimize docs and examples separately for remote and local
  install paths.

### Negative

- There is no longer one install command that serves both remote-first hosts and
  full local desktop use equally well.
- Linux users may still need standard PortAudio packages from the OS even though
  Python-level install friction is reduced.
- Callers that relied on implicit local defaults must now choose explicit local
  engines.

### Neutral

- CLI and browser examples may be install-aware and choose practical local
  defaults when the local runtimes are installed, without changing the base
  library contract.

## Enforcement

- Do not reintroduce local audio, STT, or TTS runtime dependencies into the
  base package dependency set.
- Keep local assistant dependencies behind explicit extras or install profiles.
- Keep `VoiceManager()` and plugin/server default behavior remote-first unless a
  new ADR changes that policy.
- Update installation docs and examples whenever packaging or default engine
  selection changes.

## Validation

- `tests/test_dependency_check.py`
- `tests/test_lightweight_import_boundaries.py`
- Clean-environment import/install checks for the base package
- Manual smoke of documented local install paths in release validation

## Backlog links

- `docs/backlog/completed/037_lightweight_openai_compatible_packaging.md`
- `docs/backlog/completed/038_voice_install_profiles_and_pending_defaults.md`
- `docs/backlog/completed/040_remote_first_default_and_local_extra_release.md`

## Related

- [0002_barge_in_interruption.md](./0002_barge_in_interruption.md)
- [../installation.md](../installation.md)
- [../architecture.md](../architecture.md)
