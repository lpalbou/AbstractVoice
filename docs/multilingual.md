# Multilingual support

## TTS (Piper, default)

AbstractVoice ships a small default Piper mapping for these language codes:

- `en`, `fr`, `de`, `es`, `ru`, `zh`

Programmatic usage:

```python
from abstractvoice import VoiceManager

vm = VoiceManager(language="en", allow_downloads=False)
vm.speak("Hello")

vm.set_language("fr")
vm.speak("Bonjour")
```

Offline-first note: each language requires a cached Piper voice model.

```bash
python -m abstractvoice download --piper en
python -m abstractvoice download --piper fr
```

## STT (faster-whisper, default)

STT supports many languages. You can pass a language hint when transcribing:

```python
text = vm.transcribe_file("audio.wav", language="fr")
```

## REPL

- Use `/language <lang>` to switch (`en/fr/de/es/ru/zh`).
- If a Piper model isn’t cached, the REPL will tell you to run `python -m abstractvoice download --piper <lang>`.

## AudioDiT (optional; LongCat-AudioDiT)

AudioDiT is an opt-in torch/transformers engine (`abstractvoice[audiodit]`). It uses the model `meituan-longcat/LongCat-AudioDiT-1B` and operates at **24 kHz**.

Language coverage note:

- Upstream examples and published benchmark results focus on **Chinese (ZH)** and **English (EN)**.
- You can still pass other languages (e.g. French) as plain text, but **pronunciation/intelligibility is not guaranteed** because AudioDiT does not expose a dedicated multilingual text frontend in this integration.

If you need reliable French TTS today, prefer **Piper** (`/tts_engine piper` + `/language fr`).

## OmniVoice (optional; k2-fsa/OmniVoice)

OmniVoice is an opt-in torch/transformers engine (`abstractvoice[omnivoice]`). It operates at **24 kHz** and upstream is designed for **omnilingual** TTS (600+ languages) as well as voice cloning.

Offline-first prefetch:

```bash
python -m abstractvoice download --omnivoice
# or:
abstractvoice-prefetch --omnivoice
```

Language handling note:

- When using the OmniVoice engine, AbstractVoice treats `language` as a **pass-through hint** (it does not clamp to the small Piper catalog).
- Common ISO codes like `fr`, `de`, `es`, `ru`, `zh` work; for the full set of IDs, see OmniVoice’s `LANG_IDS`.

