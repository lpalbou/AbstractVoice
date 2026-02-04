# Contributing

Thanks for your interest in contributing to AbstractVoice.

## Quick links

- User entry points: `README.md` → `docs/getting-started.md`
- Integrator contract: `docs/api.md`
- Implementation map: `docs/architecture.md`
- Internal dev notes: `docs/development.md`
- Security reports: `SECURITY.md`

## Development setup

### Requirements

- Python `>=3.8` (recommended: latest 3.12.x)
- Git

### Install (editable)

```bash
git clone https://github.com/lpalbou/abstractvoice.git
cd abstractvoice

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate

python -m pip install -U pip
python -m pip install -e ".[dev]"
```

## Running tests

Fast suite:

```bash
python -m pytest -q
```

Heavy/optional integration tests (skipped by default):

- Cloning (OpenF5): set `ABSTRACTVOICE_RUN_CLONING_TESTS=1` (also needs `pip install "abstractvoice[cloning]"`)
- Chroma: set `ABSTRACTVOICE_RUN_CHROMA_TESTS=1` (also needs `pip install "abstractvoice[chroma]"`)

## Formatting and linting (optional but recommended)

```bash
python -m black abstractvoice tests
python -m flake8 abstractvoice tests
```

## Documentation expectations

- Keep the external user flow consistent: `README.md` → `docs/getting-started.md`.
- If you change supported integrator behavior, update `docs/api.md` (source-of-truth methods live in `abstractvoice/vm/*`).
- If you change interaction semantics (voice modes / stop phrase / offline-first), update `docs/architecture.md` and add or update an ADR in `docs/adr/` when it’s a design decision.

## Pull requests

- Keep PRs focused and explain the user impact.
- Add tests for new behavior when feasible.
- Update `CHANGELOG.md` for user-visible changes.

## Reporting security issues

Do not open a public issue for suspected vulnerabilities. Follow `SECURITY.md`.
