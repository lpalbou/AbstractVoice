## Task 009: Implement Voice Cloning Adapter (optional extra)

**Status**: Planned  
**Priority**: P1  
**Depends on**: Task 008 (candidate selection)

---

## Goal

Implement a permissive-licensed voice cloning engine behind a clean API, shipped as:

```bash
pip install abstractvoice[cloning]
```

---

## Public API

- `clone_voice(reference_audio_path: str, name: str | None = None) -> str`
- `list_cloned_voices() -> list[dict]`
- `export_voice(voice_id: str, path: str) -> str`
- `import_voice(path: str) -> str`
- `speak(text: str, voice: str | None = None, ...)`

---

## REPL support

- `/clone <path>`
- `/voices` (list cloned voices)
- `/speak <text> --voice <id>`

