# Multilingual support

## TTS (remote default, Piper and Supertonic local)

`VoiceManager()` uses OpenAI remote audio by default and treats `language` as a
provider hint. The local Piper engine ships a small curated mapping for these
language codes:

- `en`, `fr`, `de`, `es`, `ru`, `zh`

Programmatic usage:

```python
from abstractvoice import VoiceManager

vm = VoiceManager(language="en", tts_engine="piper", allow_downloads=False)
vm.speak("Hello")

vm.set_language("fr")
vm.speak("Bonjour")
```

Offline-first note: each language requires a cached Piper voice model.

```bash
python -m abstractvoice download --piper en
python -m abstractvoice download --piper fr
```

Supertonic 3 is the broader fixed-profile local ONNX TTS option. It supports:

- `ar`, `bg`, `cs`, `da`, `de`, `el`, `en`, `es`, `et`, `fi`, `fr`, `hi`,
  `hr`, `hu`, `id`, `it`, `ja`, `ko`, `lt`, `lv`, `nl`, `pl`, `pt`, `ro`,
  `ru`, `sk`, `sl`, `sv`, `tr`, `uk`, `vi`

Prefetch once for all built-in styles:

```bash
python -m abstractvoice download --supertonic
```

Programmatic usage:

```python
from abstractvoice import VoiceManager

vm = VoiceManager(language="fr", tts_engine="supertonic", allow_downloads=False)
vm.set_profile("F1")
vm.speak("Bonjour")
```

## STT (OpenAI default, faster-whisper local)

OpenAI remote transcription is the default. Local faster-whisper supports many
languages when selected explicitly. You can pass a language hint when
transcribing:

```python
from abstractvoice import VoiceManager

vm = VoiceManager(tts_engine="piper", stt_engine="faster_whisper")
text = vm.transcribe_file("audio.wav", language="fr")
```

## REPL

- Use `/language <lang>` to switch (`en/fr/de/es/ru/zh` for Piper; the
  Supertonic list above when `tts_engine=supertonic`).
- If a Piper model isn’t cached, the REPL will tell you to run `python -m abstractvoice download --piper <lang>`.
- If Supertonic isn’t cached, run `python -m abstractvoice download --supertonic`
  or `/tts_download supertonic`.

## AudioDiT (optional; LongCat-AudioDiT)

AudioDiT is an opt-in torch/transformers engine (`abstractvoice[audiodit]`). It uses the model `meituan-longcat/LongCat-AudioDiT-1B` and operates at **24 kHz**.

Language coverage note:

- Upstream examples and published benchmark results focus on **Chinese (ZH)** and **English (EN)**.
- You can still pass other languages (e.g. French) as plain text, but **pronunciation/intelligibility is not guaranteed** because AudioDiT does not expose a dedicated multilingual text frontend in this integration.

If you need reliable French TTS today, start with **Piper** for the smallest
local footprint or **Supertonic** for broader fixed-profile multilingual TTS.

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
