## Task 040: Remote-first default install and local stack extra for 0.9.1

**Date**: 2026-05-08  
**Status**: Completed  
**Priority**: P1

### Context

Task 038 made the base package lightweight but kept direct `VoiceManager()`
local-first, and task 039 added the AbstractCore voice catalog surface on top of
remote plugin defaults. Release review accepted the plugin work but identified a
remaining persona split: the default library constructor still selected local
`auto` engines while the plugin selected remote OpenAI engines.

For `0.9.1`, the release decision is to make the default install remote-first
throughout AbstractVoice. Local inference, microphone listening dependencies,
AEC, and local cloning/TTS engines move behind one explicit handle:
`abstractvoice[local]`.

### Decision

- `VoiceManager()` defaults to `tts_engine="openai"` and `stt_engine="openai"`.
- `auto` resolves to OpenAI remote audio for both TTS and STT.
- `abstractvoice[local]` installs the full local stack, including Piper,
  faster-whisper, audio I/O, AEC where supported, and optional local
  cloning/TTS engines gated by Python-version markers.
- Legacy compatibility extras are removed without aliases:
  `voice`, `voice-full`, `local-tts`, `local-stt`, `core-stt`, `audio-only`,
  `legacy-stt`, `all`, and the `web-*` engine bundles.
- The legacy `abstractvoice.stt.Transcriber` / `openai-whisper` fallback is
  removed from the supported runtime.

### Implementation

- Updated package metadata in `pyproject.toml`.
- Updated `VoiceManager`, TTS/STT adapter selection, CLI/web defaults, install
  hints, and dependency diagnostics.
- Updated tests for remote-first defaults, removed extras, and lightweight
  import boundaries.
- Updated README, installation, getting-started, API, architecture, FAQ, REPL,
  dependency docs, ADR 0001, `llms.txt`, and `llms-full.txt`.
- Bumped the release version to `0.9.1` and moved release notes into
  `CHANGELOG.md`.

### Validation

- `python -m pytest -q tests/test_dependency_check.py tests/test_lightweight_import_boundaries.py tests/test_remote_openai_compatible_adapters.py tests/test_abstractcore_plugin.py`
- `python -m pytest -q`
- `git diff --check`
- `python -m build --sdist --wheel --outdir /tmp/abstractvoice-build-0.9.1`
- `python -m twine check /tmp/abstractvoice-build-0.9.1/*`
