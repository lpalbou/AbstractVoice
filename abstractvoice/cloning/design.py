"""Design a voice from a written description and keep it as a reusable cloned voice.

Qwen3-TTS VoiceDesign renders a reference in ONE generation: a short warm-up
sentence, then the reference text. VoiceDesign settles into its voice over the
first seconds of a generation, so the warm-up is cut off at the pause that
follows it, and the remaining clip (the reference text alone) is cloned with a
Qwen3-TTS Base checkpoint. The result is a fixed voice identity, unlike a
description used at synthesis time, which samples a slightly different voice on
every call. See docs/voice-pack.md for the same method applied to the built-in voices.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

DESIGN_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
CLONING_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

# One warm-up sentence and one reference text per Qwen3-TTS language.
WARMUPS = {
    "en": "This is a short introduction, so that my voice can settle before the main part begins.",
    "fr": "Ceci est une courte introduction, pour que ma voix puisse se poser avant de commencer.",
    "es": "Esta es una breve introducción, para que mi voz se asiente antes de empezar.",
    "de": "Dies ist eine kurze Einleitung, damit sich meine Stimme einpendeln kann, bevor es losgeht.",
    "it": "Questa è una breve introduzione, perché la mia voce possa assestarsi prima di cominciare.",
    "pt": "Esta é uma breve introdução, para que a minha voz se estabilize antes de começar.",
    "ru": "Это короткое вступление, чтобы мой голос успел настроиться перед основной частью.",
    "zh": "这是一段简短的开场白，让我的声音在正式开始之前稳定下来。",
    "ja": "これは短い前置きです。本題に入る前に、声を落ち着かせるためのものです。",
    "ko": "이것은 짧은 도입부입니다. 본론에 들어가기 전에 목소리를 가다듬기 위한 것입니다.",
}
TEXTS = {
    "en": "Hello, and welcome. I'm really glad you're here. Let me tell you a little about today: we'll start with the essentials, take our time with the details, and finish with a few nice surprises.",
    "fr": "Bonjour et bienvenue. Je suis vraiment ravi de vous accueillir. Laissez-moi vous présenter la journée : nous commencerons par l'essentiel, nous prendrons le temps des détails, et nous terminerons par quelques belles surprises.",
    "es": "Hola y bienvenidos. Me alegra mucho que estéis aquí. Os cuento un poco sobre el día de hoy: empezaremos por lo esencial, nos tomaremos nuestro tiempo con los detalles y terminaremos con algunas sorpresas.",
    "de": "Hallo und herzlich willkommen. Ich freue mich sehr, dass Sie hier sind. Lassen Sie mich kurz erzählen, was heute ansteht: Wir beginnen mit dem Wesentlichen, nehmen uns Zeit für die Details und enden mit ein paar schönen Überraschungen.",
    "it": "Ciao e benvenuti. Sono davvero felice che siate qui. Vi racconto un po' della giornata di oggi: cominceremo dall'essenziale, ci prenderemo il tempo per i dettagli e finiremo con qualche bella sorpresa.",
    "pt": "Olá e bem-vindos. Fico muito feliz por estarem aqui. Deixem-me contar um pouco sobre o dia de hoje: começamos pelo essencial, dedicamos tempo aos detalhes e terminamos com algumas boas surpresas.",
    "ru": "Здравствуйте и добро пожаловать. Я очень рад, что вы здесь. Позвольте немного рассказать о сегодняшнем дне: мы начнём с самого главного, не спеша разберём детали и закончим несколькими приятными сюрпризами.",
    "zh": "大家好，欢迎光临。很高兴你们能来。让我简单介绍一下今天的安排：我们先从最重要的内容开始，然后慢慢讲解细节，最后还会有一些小惊喜。",
    "ja": "こんにちは、ようこそ。お越しいただき本当にうれしいです。今日の流れを少しご紹介します。まず大切なことから始めて、細かい点をじっくり見て、最後にいくつか素敵な驚きをご用意しています。",
    "ko": "안녕하세요, 환영합니다. 와 주셔서 정말 기쁩니다. 오늘 일정을 간단히 소개해 드릴게요. 가장 중요한 것부터 시작해서 세부 사항을 천천히 살펴보고, 마지막에는 몇 가지 즐거운 깜짝 선물이 있습니다.",
}


def find_warmup_cut(audio: np.ndarray, sample_rate: int, *, warmup_share: float) -> Optional[int]:
    """Sample index of the pause that ends the warm-up, or None.

    ``warmup_share`` is the warm-up's share of the text; the pause nearest to that
    share of the audio's duration (within 0.6x-1.6x of it) is chosen.
    """
    a = np.asarray(audio, dtype=np.float32).reshape(-1)
    sr = int(sample_rate)
    frame, hop = int(0.02 * sr), int(0.01 * sr)
    if len(a) < frame * 10:
        return None
    rms = np.array([np.sqrt(np.mean(a[i:i + frame] ** 2)) for i in range(0, len(a) - frame, hop)])
    threshold = max(0.006, 0.12 * float(np.percentile(rms, 90)))
    silent = rms < threshold
    runs, start = [], None
    for i, s in enumerate(np.append(silent, False)):
        if s and start is None:
            start = i
        elif not s and start is not None:
            if (i - start) * hop >= 0.15 * sr:
                runs.append((start, i))
            start = None
    expected = warmup_share * len(a)
    best = None
    for s, e in runs:
        center = (s + e) // 2 * hop + frame // 2
        if 0.6 * expected <= center <= 1.6 * expected:
            if best is None or abs(center - expected) < abs(best - expected):
                best = center
    return best


def render_designed_reference(
    description: str,
    *,
    language: str = "en",
    text: Optional[str] = None,
    model_id: str = DESIGN_MODEL_ID,
    allow_downloads: bool = True,
    attempts: int = 3,
) -> Tuple[np.ndarray, int, str]:
    """Render ``(audio, sample_rate, reference_text)``: the settled voice reading ``text``.

    Raises ValueError for an empty description or an unsupported language without
    ``text``, RuntimeError when no attempt yields a clean cut.
    """
    from ..adapters.tts_qwen3_tts import _qwen_language
    from ..qwen3_tts.runtime import Qwen3TTSRuntime, one_shot_max_new_tokens

    if not str(description or "").strip():
        raise ValueError("a voice description is required")
    lang = str(language or "en").strip().lower()
    if lang not in WARMUPS:
        raise ValueError(f"design_voice supports the Qwen3-TTS languages {', '.join(WARMUPS)} (got {language!r})")
    reference_text = str(text).strip() if text and str(text).strip() else TEXTS[lang]
    warmup = WARMUPS[lang]
    full = f"{warmup} {reference_text}"
    runtime = Qwen3TTSRuntime(model_id=model_id, allow_downloads=bool(allow_downloads))
    try:
        for _ in range(max(1, int(attempts))):
            audio, sr = runtime.synthesize_voice_design(
                full,
                instruct=str(description),
                language=_qwen_language(lang),
                max_new_tokens=one_shot_max_new_tokens(full),
            )
            cut = find_warmup_cut(audio, sr, warmup_share=len(warmup) / len(full))
            if cut is None:
                continue
            ref = np.asarray(audio[cut:], dtype=np.float32).copy()
            n = int(0.01 * sr)
            ref[:n] *= np.linspace(0.0, 1.0, n, dtype=np.float32)  # 10 ms fade-in at the cut
            if len(ref) / sr >= 4.0:
                return ref, int(sr), reference_text
        raise RuntimeError("could not separate the warm-up from the reference; try again or rephrase the description")
    finally:
        runtime.unload()
