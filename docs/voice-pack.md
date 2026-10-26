# Built-in voices

AbstractVoice ships 70 ready-made voices for Qwen3-TTS: women and men in American and
British English, French, Spanish, German and Italian, including a deep, professional bass male voice in
every language. They behave like voices you cloned yourself (same ids, same calls, same listings), are
available offline once the model is cached, and are read-only.

This page covers what you need to use them, the full list, and how they were made. For cloning your own
voices see [API: voice cloning](api.md#voice-cloning-optional-heavy); for licensing see
[Voices and licenses](voices-and-licenses.md#built-in-voices).

## Requirements

- `pip install "abstractvoice[qwen3-tts]"` (Python 3.10+; on Apple silicon it runs on the Apple GPU through
  MLX, elsewhere on PyTorch).
- The `Qwen/Qwen3-TTS-12Hz-1.7B-Base` checkpoint, which renders every built-in voice:
  `python -m abstractvoice download --qwen3-tts Qwen/Qwen3-TTS-12Hz-1.7B-Base`
  (or allow downloads and let the first use fetch it).

The package itself only carries each voice's clone prompt (its reference codec codes and speaker
embedding, about 12 KB per voice); no reference audio is shipped.

## Use them

The voice id is all you need. Any voice can read any of the ten Qwen3-TTS languages; it sounds most
natural in its own.

```python
from abstractvoice import VoiceManager

vm = VoiceManager(cloning_engine="qwen3-tts")
wav = vm.speak_to_bytes("Bonjour et bienvenue.", voice="fr-female-camille")
vm.speak_to_file("Good evening.", "news.wav", voice="en-gb-male-oliver", one_shot=True)
for chunk, sr in vm.speak_to_audio_chunks("A longer reply, streamed as it is generated.", voice="en-us-male-harrison"):
    ...
vm.speak("Hallo zusammen!", voice="de-female-anna")
[v["voice_id"] for v in vm.list_cloned_voices() if v["meta"].get("builtin")]
```

`one_shot=True` renders a whole text as one generation, so a take keeps one voice from start to end
(see [API: TTS](api.md#tts-text--audio)).

Command line and REPL:

```bash
abstractvoice --prompt "Bonjour et bienvenue." --voice fr-female-camille --output camille.wav --one-shot
abstractvoice                 # then: /voices clones   and   /voices clone fr-female-camille
```

AbstractCore (the voice capability plugin) lists them with your own clones and speaks with them by id:

```python
llm.voice.voice_catalog(provider="qwen3-tts")["cloned_voices"]  # built-in voices: tags["builtin"] is True
llm.voice.tts("Gracias por esperar.", voice="es-female-lucia", one_shot=True)
llm.voice.load_resident_model({"task": "tts", "provider": "cloned", "model": "qwen3-tts",
                               "options": {"voice": "it-male-vittorio"}})   # warm it at load
```

Over HTTP, AbstractCore Server lists them in `GET /v1/audio/voices` and accepts the id as `voice` in
`POST /v1/audio/speech`.

Each voice carries its profile: `builtin`, `nationality`, `language`, `language_code`, `sex`, `style`
(`standard` or `bass`), `description` and `model_id`. `VoiceManager.list_cloned_voices()` returns them
under `meta`; the AbstractCore plugin's catalog entries (`voice_catalog()`, `list_cloned_voices()`,
`GET /v1/audio/voices`) also expose them in `tags`, so clients can filter by language, sex or style.

Built-in voices are read-only: renaming, editing the transcript, exporting or deleting one raises
`ValueError`. A clone of your own with the same id takes precedence over the built-in voice.

## The voices

### American (English (United States))

| Voice id | Name | Sex | Style | Description |
|---|---|---|---|---|
| `en-us-female-emily` | Emily | female | standard | A friendly American woman in her thirties with a clear General American accent, warm and upbeat, natural conversational pace. |
| `en-us-female-grace` | Grace | female | standard | A poised American woman in her forties with a crisp, professional voice, clear diction, like a documentary narrator. |
| `en-us-female-linda` | Linda | female | standard | A calm, mature American woman in her fifties with a smooth, reassuring voice, like a public radio host. |
| `en-us-female-madison` | Madison | female | standard | A bright, energetic American woman in her early twenties, cheerful and expressive, slightly fast-paced, Californian accent. |
| `en-us-female-ruth` | Ruth | female | standard | A gentle elderly American woman in her seventies, kind and soft-spoken, like a loving grandmother reading a bedtime story. |
| `en-us-male-grant` | Grant | male | bass | A very low, rich bass American male voice, strong and commanding, like a movie trailer narrator. Deep chest resonance, measured pace. |
| `en-us-male-harrison` | Harrison | male | bass | A deep, powerful American male voice with a resonant bass tone, solid and authoritative, like a professional broadcast announcer. Low pitch, slow and confident. |
| `en-us-male-wade` | Wade | male | bass | A warm, deep bass American male voice in his fifties, steady and trustworthy, like a corporate presentation voice-over. Low and smooth. |
| `en-us-male-jake` | Jake | male | standard | A relaxed, friendly American man in his thirties with a General American accent, casual and approachable. |
| `en-us-male-marcus` | Marcus | male | standard | An enthusiastic young American man in his twenties with a clearly masculine, mid-low voice, upbeat and dynamic, like a podcast host. |
| `en-us-male-michael` | Michael | male | standard | A confident American man in his forties with a rich baritone voice, articulate and steady, like a television news anchor. |
| `en-us-male-walter` | Walter | male | standard | An older American man in his sixties with a deep, slightly gravelly voice, slow and thoughtful, like a storyteller by a campfire. |

### British (English (United Kingdom))

| Voice id | Name | Sex | Style | Description |
|---|---|---|---|---|
| `en-gb-female-charlotte` | Charlotte | female | standard | A British woman in her thirties with an elegant Received Pronunciation accent, warm, composed and articulate. |
| `en-gb-female-eleanor` | Eleanor | female | standard | A British woman in her forties with a calm, authoritative Received Pronunciation accent, like an audiobook narrator. |
| `en-gb-female-isla` | Isla | female | standard | A Scottish woman in her thirties from Edinburgh with a soft, melodic Scottish accent, warm and friendly. |
| `en-gb-female-margaret` | Margaret | female | standard | A refined elderly British woman in her seventies with a gentle, cultured Received Pronunciation accent. |
| `en-gb-female-olivia` | Olivia | female | standard | A young British woman in her twenties from London, lively and friendly, with a modern southern English accent. |
| `en-gb-male-alistair` | Alistair | male | bass | A deep, powerful British male voice with a resonant bass tone and a refined Received Pronunciation accent, authoritative, like a professional broadcast announcer. Low pitch, slow and confident. |
| `en-gb-male-graham` | Graham | male | bass | A very low, rich bass British male voice, strong and commanding, like a film trailer narrator, Received Pronunciation. Deep chest resonance. |
| `en-gb-male-rupert` | Rupert | male | bass | A warm, deep bass British male voice in his fifties, steady and trustworthy, like a documentary voice-over. Low and smooth. |
| `en-gb-male-edward` | Edward | male | standard | An older British man in his sixties with a deep, rich voice and a Received Pronunciation accent, like a nature documentary narrator. |
| `en-gb-male-harry` | Harry | male | standard | A young British man in his twenties from London, relaxed and cheerful, with a casual southern English accent. |
| `en-gb-male-liam` | Liam | male | standard | A Northern English man in his thirties from Manchester, down-to-earth and friendly, with a clear Mancunian accent. |
| `en-gb-male-oliver` | Oliver | male | standard | A middle-aged British man with a refined Received Pronunciation accent, polite and measured, like a BBC presenter. |

### French (French)

| Voice id | Name | Sex | Style | Description |
|---|---|---|---|---|
| `fr-female-camille` | Camille | female | standard | Une femme française d'une trentaine d'années, voix douce, claire et naturelle, accent parisien. |
| `fr-female-elise` | Élise | female | standard | Une femme française d'une quarantaine d'années, voix féminine douce, posée et élégante, comme une présentatrice de radio. |
| `fr-female-juliette` | Juliette | female | standard | Une femme française d'une trentaine d'années, voix féminine claire et mélodieuse, chaleureuse, accent parisien. |
| `fr-female-lea` | Léa | female | standard | Une jeune femme française d'une vingtaine d'années, voix pétillante et enjouée, débit un peu rapide. |
| `fr-female-margaux` | Margaux | female | standard | Une femme française d'une quarantaine d'années, voix calme et précise, comme une narratrice de documentaire. |
| `fr-male-bernard` | Bernard | male | bass | Une voix d'homme française de basse, profonde, riche et imposante, comme la voix d'une bande-annonce de cinéma. |
| `fr-male-rene` | René | male | bass | Une voix d'homme française grave et chaleureuse, la cinquantaine, posée et rassurante, comme une voix off professionnelle. |
| `fr-male-antoine` | Antoine | male | standard | Un homme français d'une quarantaine d'années, voix de baryton posée et élégante, comme un présentateur de journal télévisé. |
| `fr-male-jean-pierre` | Jean-Pierre | male | standard | Un homme français d'une soixantaine d'années, voix grave et profonde, débit lent, comme un conteur. |
| `fr-male-julien` | Julien | male | standard | Un homme français d'une trentaine d'années, voix chaleureuse et détendue, accent parisien. |
| `fr-male-thomas` | Thomas | male | standard | Un jeune homme français d'une vingtaine d'années, voix dynamique et sympathique. |

### Spanish (Spanish (Spain))

| Voice id | Name | Sex | Style | Description |
|---|---|---|---|---|
| `es-female-carmen` | Carmen | female | standard | Una mujer española de unos sesenta años, voz serena, cariñosa y tranquilizadora. |
| `es-female-elena` | Elena | female | standard | Una locutora de radio española de unos cuarenta años, voz profesional, nítida y segura. |
| `es-female-isabel` | Isabel | female | standard | Una mujer española de Andalucía, de unos treinta años, voz alegre y melodiosa con acento andaluz suave. |
| `es-female-lucia` | Lucía | female | standard | Una mujer española de Madrid, de unos treinta años, voz cálida, clara y natural. |
| `es-female-sofia` | Sofía | female | standard | Una joven española de unos veinte años, voz alegre y enérgica, ritmo algo rápido. |
| `es-male-alfonso` | Alfonso | male | bass | Una voz masculina española muy grave y potente, timbre de bajo resonante, sólida y autoritaria, como un locutor profesional. Ritmo lento y seguro. |
| `es-male-fernando` | Fernando | male | bass | Un hombre español de unos cincuenta años, voz masculina muy grave y cálida, de bajo, serena y de confianza, como una voz en off profesional. |
| `es-male-ramon` | Ramón | male | bass | Una voz masculina española de bajo, profunda, rica e imponente, como la voz de un tráiler de cine. |
| `es-male-antonio` | Antonio | male | standard | Un hombre español de unos sesenta años, voz grave y profunda, ritmo pausado, como un narrador de cuentos. |
| `es-male-diego` | Diego | male | standard | Un hombre español de unos cuarenta años, voz de barítono firme y elegante, como un presentador de noticias. |
| `es-male-javier` | Javier | male | standard | Un hombre español de Madrid, de unos treinta y cinco años, voz cálida y cercana. |
| `es-male-pablo` | Pablo | male | standard | Un joven español de unos veinticinco años, voz masculina, dinámica y simpática. |

### German (German)

| Voice id | Name | Sex | Style | Description |
|---|---|---|---|---|
| `de-female-anna` | Anna | female | standard | Eine deutsche Frau Anfang dreißig mit heller, weiblicher Stimme, warm, klar und natürlich, hochdeutsche Aussprache. |
| `de-female-greta` | Greta | female | standard | Eine deutsche Frau um die vierzig mit ruhiger, weiblicher Stimme, präzise und angenehm, wie eine Dokumentarfilm-Erzählerin. |
| `de-female-lena` | Lena | female | standard | Eine junge deutsche Frau Mitte zwanzig, fröhliche und lebhafte Stimme, etwas schnelles Sprechtempo. |
| `de-female-sophie` | Sophie | female | standard | A cheerful German woman in her thirties with a bright, energetic, clearly feminine voice, like a radio host, speaking German. |
| `de-female-ursula` | Ursula | female | standard | Eine deutsche Frau um die sechzig, ruhige, freundliche und vertrauensvolle Stimme. |
| `de-male-dieter` | Dieter | male | bass | Eine tiefe, volle deutsche Bassstimme, stark und eindrucksvoll, wie die Stimme eines Kinotrailers. |
| `de-male-horst` | Horst | male | bass | Eine warme, tiefe deutsche Männerstimme Mitte fünfzig, ruhig und vertrauenswürdig, wie ein professioneller Off-Sprecher. |
| `de-male-wolfgang` | Wolfgang | male | bass | Eine sehr tiefe, kraftvolle deutsche Männerstimme mit resonantem Bass, solide und souverän, wie ein professioneller Sprecher. Langsames, sicheres Tempo. |
| `de-male-jonas` | Jonas | male | standard | Ein junger deutscher Mann Mitte zwanzig, dynamische und sympathische Stimme. |
| `de-male-klaus` | Klaus | male | standard | Ein deutscher Mann um die sechzig, tiefe und sonore Stimme, langsames Sprechtempo, wie ein Erzähler. |
| `de-male-lukas` | Lukas | male | standard | Ein deutscher Mann Mitte dreißig, warme und entspannte Stimme, klares Hochdeutsch. |
| `de-male-maximilian` | Maximilian | male | standard | Ein deutscher Mann um die vierzig, souveräne Baritonstimme, wie ein Nachrichtensprecher. |

### Italian (Italian)

| Voice id | Name | Sex | Style | Description |
|---|---|---|---|---|
| `it-female-chiara` | Chiara | female | standard | Una giovane donna italiana sui vent'anni, voce allegra e vivace, ritmo un po' veloce. |
| `it-female-francesca` | Francesca | female | standard | Una conduttrice radiofonica italiana sui quarant'anni, voce professionale, nitida e sorridente. |
| `it-female-giulia` | Giulia | female | standard | Una donna italiana sui trent'anni, voce calda, chiara e naturale, accento neutro. |
| `it-female-lucia` | Lucia | female | standard | Una donna italiana sui sessant'anni, voce calma, affettuosa e rassicurante. |
| `it-female-rosa` | Rosa | female | standard | Una donna italiana di Napoli sui trent'anni, voce solare e melodiosa con un leggero accento napoletano. |
| `it-male-enzo` | Enzo | male | bass | Una voce maschile italiana grave e calda, sulla cinquantina, pacata e rassicurante, come una voce fuori campo professionale. |
| `it-male-massimo` | Massimo | male | bass | Una voce maschile italiana di basso, profonda, ricca e imponente, come la voce di un trailer cinematografico. |
| `it-male-vittorio` | Vittorio | male | bass | Una voce maschile italiana molto grave e potente, timbro di basso risonante, solida e autorevole, come un annunciatore professionista. Ritmo lento e sicuro. |
| `it-male-alessandro` | Alessandro | male | standard | Un uomo italiano sui quarant'anni, voce baritonale elegante e sicura, come un conduttore di telegiornale. |
| `it-male-giuseppe` | Giuseppe | male | standard | Un uomo italiano sui sessant'anni, voce profonda e grave, ritmo lento, come un narratore di storie. |
| `it-male-luca` | Luca | male | standard | Un giovane uomo italiano sui vent'anni, voce dinamica e simpatica. |

## How they were made

1. **Designed.** Each voice was described in words to the Qwen3-TTS 1.7B VoiceDesign model (the
   descriptions above), in the target language, stating the voice's gender explicitly.
2. **One generation, settled.** Each reference was rendered with `one_shot=True` as a short warm-up
   sentence followed by the reference text. VoiceDesign settles into its voice over the first seconds
   of a generation, so the warm-up was cut off at the pause before the reference text, located with
   Whisper word timestamps.
3. **Checked.** Of three takes, only those transcribed in the right language with at most 9 % word
   errors were kept, and the steadiest one (speaker-embedding similarity of every 2.5 s window to the
   clip's average voice) was chosen. Gender was cross-checked by pitch and a speaker-embedding vote.
4. **Listened to.** Every voice was accepted by ear before it was cloned.
5. **Cloned.** `tools/voice_pack/build.py` encoded each reference once with the 1.7B Base checkpoint
   (PyTorch reference implementation) and stored the clone prompt in
   `abstractvoice/assets/voices/qwen3_tts/` (`voices.json` and `prompts/<voice-id>.npz`).
6. **Verified.** Every voice then spoke a new sentence in its language through the AbstractCore plugin,
   loaded from the pack alone, and was checked for words, language and likeness to its reference.

To add or rebuild voices, put validated references and a `profiles.json` (id, file, name, nationality,
language, language_code, sex, style, description, reference_text) in a folder and run
`python tools/voice_pack/build.py <folder>` from a repository checkout.
