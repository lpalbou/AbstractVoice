# Documentation

The core documentation set is:

- `README.md`
- `docs/getting-started.md`
- `docs/api.md`
- `docs/architecture.md`
- `docs/faq.md`

Use those five files for the package overview, first run, supported API,
implementation map, cache/history reset, and common troubleshooting.

If you’re using AbstractVoice as part of the AbstractFramework ecosystem
(AbstractCore / AbstractRuntime), see the integration notes in `README.md` and
the “Integrations” section in `docs/api.md`. In that mode, AbstractCore owns
the server/API surface; AbstractVoice is the capability plugin that backs
voice/audio operations such as `/v1/audio/speech` and
`/v1/audio/transcriptions`.

For a browser smoke test, install `abstractvoice[web]` and run
`OPENAI_API_KEY=... abstractvoice web`. That web UI is a small `VoiceManager` example with
assistant/user voice selection, browser voice cloning from uploaded or recorded
reference audio, and an example-only local LLM dialogue bridge, not the
production API server. Compose extras such as `abstractvoice[web,supertonic]`
or use `abstractvoice[all-apple]` / `abstractvoice[all-gpu]` when you want
local engines in the same install.

## User-facing docs

- **FAQ**: `docs/faq.md`
- **Known issues**: `docs/known-issues.md`
- **Install & troubleshooting**: `docs/installation.md`
- **REPL guide (voice assistant)**: `docs/repl_guide.md`
- **Multilingual**: `docs/multilingual.md`
- **Voice/model management**: `docs/model-management.md`
- **Dependencies & model sources**: `docs/dependencies.md`
- **Licensing notes for models/voices**: `docs/voices-and-licenses.md`
- **Benchmarks (TTS-only)**: `docs/benchmark.md`
- **Preload benchmarks (local TTS/STT)**: `examples/bench_preload_local_models.py`

## For integrators

- **Stable API surface**: `docs/api.md`
- **Architecture (implementation map)**: `docs/architecture.md`
- **ADRs (design decisions)**: `docs/adr/`

## Project

- **Changelog**: `CHANGELOG.md`
- **Contributing**: `CONTRIBUTING.md`
- **Bug reports**: `.github/ISSUE_TEMPLATE/bug_report.yml`
- **Security**: `SECURITY.md`
- **Acknowledgments**: `ACKNOWLEDGMENTS.md`
- **License**: `LICENSE`

## Internal / historical (not an API contract)

These documents are useful for maintainers, but may describe past work or future plans:

- `docs/backlog/` (planning + task notes)
- `docs/reports/` (status and implementation reports)
- `docs/voice_cloning_2026.md` (research notes; not a promise of product behavior)
