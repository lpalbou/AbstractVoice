# Technical Documentation

This directory contains technical documentation for AbstractVoice developers and advanced users.

## 📖 Available Documentation

### [overview.md](overview.md)
**High-level orientation**
- What AbstractVoice is for
- Supported usage modes (local/headless/REPL)

### [acronyms.md](acronyms.md)
**Shared vocabulary**
- Definitions for common acronyms used across docs (TTS, STT, VAD, VM, MM, …)

### [public_api.md](public_api.md)
**Supported integrator API contract**
- The small set of front-facing methods we want integrators to rely on

### [repl_guide.md](repl_guide.md)
**How to use the REPL as a general voice assistant**
- Smoke test checklist
- Common commands

### [getting_started.md](getting_started.md)
**Local developer setup + testing**
- venv setup
- run tests
- run the local assistant

### [adr/](adr/)
**Architecture Decision Records**
- design constraints and trade-offs captured as ADRs

### [architecture.md](architecture.md)
**Complete technical architecture guide**
- Component overview and communication patterns
- Threading model and thread safety
- Implementation details of immediate pause/resume functionality
- Performance characteristics and optimization

### [development.md](development.md)
**Development insights and best practices**
- State-of-the-art TTS implementation details
- Long text synthesis challenges and solutions
- Model selection and quality considerations
- Performance optimization techniques
- Error handling and fallback strategies

### [model-management.md](model-management.md) ⭐ **NEW v0.4.0**
**Complete model management guide**
- Essential vs language-specific models
- Programmatic APIs for third-party integration
- CLI commands for model download and management
- Voice selection and cache management
- Integration patterns for libraries and web APIs

### [voice_cloning_2026.md](voice_cloning_2026.md)
**Investigation report**
- Permissive (MIT/Apache) voice cloning options as of Jan 2026

## 🎯 Quick Links

**For Users**: See [../README.md](../README.md) for installation, usage, and API reference

**For Contributors**: See [../CONTRIBUTING.md](../CONTRIBUTING.md) for development setup and guidelines

**For Version History**: See [../CHANGELOG.md](../CHANGELOG.md) for release notes and changes