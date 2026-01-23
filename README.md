## AbstractVoice

A modular Python library for **voice I/O** around AI applications.

- **TTS (default)**: Piper (cross-platform, no system deps)
- **STT (default)**: faster-whisper
- **Local assistant**: `listen()` + `speak()` with playback/listening control
- **Headless/server**: `speak_to_bytes()` / `speak_to_file()` and `transcribe_*`

> AbstractVoice will ultimately be integrated as the voice modality of AbstractFramework.  
> An OpenAI-compatible voice endpoint is an optional demo/integration layer (see backlog).

---

## Install

```bash
pip install abstractvoice
```

Optional extras (legacy engines / extra tooling):

```bash
pip install "abstractvoice[all]"
```

---

## Quick smoke tests

### REPL (fastest end-to-end)

```bash
abstractvoice cli --debug
```

See `docs/repl_guide.md`.

### Minimal Python

```python
from abstractvoice import VoiceManager

vm = VoiceManager()
vm.speak("Hello! This is AbstractVoice.")
```

---

## Public API (stable surface)

See `docs/public_api.md` for the supported integrator contract.

At a glance:
- **TTS**: `speak()`, `stop_speaking()`, `pause_speaking()`, `resume_speaking()`, `speak_to_bytes()`, `speak_to_file()`
- **STT**: `transcribe_file()`, `transcribe_from_bytes()`
- **Mic**: `listen()`, `stop_listening()`, `pause_listening()`, `resume_listening()`

---

## Documentation (minimal set)

- **Orientation**: `docs/overview.md`
- **Acronyms**: `docs/acronyms.md`
- **Public API**: `docs/public_api.md`
- **REPL guide**: `docs/repl_guide.md`
- **Install troubleshooting**: `docs/installation.md`
- **Multilingual support**: `docs/multilingual.md`
- **Architecture (internal)**: `docs/architecture.md` + `docs/adr/`
- **Legacy model management (Coqui)**: `docs/model-management.md`
- **Voice cloning investigation**: `docs/voice_cloning_2026.md`
- **Licensing notes**: `docs/voices-and-licenses.md`

---

## License

MIT. See `LICENSE`.

