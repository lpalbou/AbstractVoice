# Known Issues

GitHub Issues is the canonical tracker for active bugs. This page is the
release-facing summary of issues users should know about before selecting an
engine or workflow.

Recommended labels for GitHub:

- `bug`
- `known-issue`
- `engine:audiodit`
- `engine:omnivoice`
- `release:0.8.1`
- `priority:normal` or `priority:high`

Use the bug report template in `.github/ISSUE_TEMPLATE/bug_report.yml` for new
reports so every issue includes version, engine, reproduction steps, environment,
and artifacts.

When a GitHub issue exists, link it from the matching entry below. When the bug
is fixed, close the GitHub issue, move the resolution to `CHANGELOG.md`, and
remove the active entry here.

## Active

### AV-KNOWN-001: AudioDiT direct/base TTS can sound distorted

Status: active in `0.8.1`

Affected path:

- `VoiceManager(tts_engine="audiodit").speak(...)`
- `VoiceManager(tts_engine="audiodit").speak_to_bytes(...)`
- REPL: `/tts_engine audiodit` followed by `/speak ...`

Observed behavior:

- Direct AudioDiT speech can include audible distortion in this release.
- AudioDiT cloning is not blocked by this note and remains the better-validated
  AudioDiT path.

Recommended workaround:

- Use Piper for reliable direct local TTS.
- Use AudioDiT cloning only after validating the target voice and hardware.
- Use OmniVoice or OpenF5/Chroma where those engines match the voice-cloning
  requirement better.

Planned follow-up:

- Add a small reproducible direct-TTS quality test sample.
- Compare base TTS settings against upstream examples and current defaults.
- Track a before/after audio artifact when the issue is fixed.

### AV-KNOWN-002: OmniVoice preset voice profiles need more stability work

Status: active in `0.8.1`

Affected path:

- REPL: `/tts_engine omnivoice` with `/profile ...`
- `VoiceManager(tts_engine="omnivoice").set_profile(...)`
- OmniVoice designed voices based primarily on `instruct`, seed, and sampling
  settings

Observed behavior:

- Designed voice identity can still vary across turns, devices, torch dtypes, or
  model/runtime versions.
- Preset profiles are useful for experimentation, but should not yet be treated
  as fully stable production speaker identities.

Recommended workaround:

- Prefer profiles that build a persistent prompt cache when available.
- For stronger persistence, create a cloned voice from reference audio and use
  `/clone_export` / `/clone_import` to move it between machines.
- Keep the same OmniVoice model snapshot, torch stack, device, dtype, seed, and
  generation settings when comparing results.

Planned follow-up:

- Curate and validate stable preset profiles with objective sample artifacts.
- Document which profile settings are deterministic enough per device class.
- Promote validated profiles from demo/experimental to recommended presets.
