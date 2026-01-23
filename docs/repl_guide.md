# REPL Guide (CLI Voice Assistant)

The REPL is the quickest way to validate your installation end‑to‑end.

## Start

```bash
abstractvoice cli --debug
```

## Quick smoke test checklist

### 1) TTS works

- Type a short message to the assistant (it will call your configured LLM endpoint).
- Or use `/speak <text>` to test TTS without calling the LLM.
- Use:
  - `/pause`
  - `/resume`
  - `/stop`

### 2) Language switching (Piper path)

Try:
- `/language fr`
- `/language de`
- `/language es`
- `/language ru`
- `/language zh`

### 3) Disable/enable TTS

- `/tts off` (text‑only)
- `/tts on` (re‑enables voice features)

### 4) Engine selection

- `/tts_engine piper`
- `/tts_engine vits` (legacy, may require extra deps)
- `/stt_engine faster_whisper`

### 4) Voice catalog (legacy, optional)

- `/setvoice` lists voices.
- `/setvoice fr.css10_vits` attempts to switch to a specific legacy model.

This is optional and may require additional dependencies (see README install extras).

## Troubleshooting

- If the REPL can’t connect to your LLM server, ensure it’s running and the `--api` URL is correct.

