"""Design a voice from a written description and keep it as a reusable cloned voice.

Qwen3-TTS VoiceDesign renders a reference in ONE generation: a short warm-up
sentence, then the reference text. VoiceDesign settles into its voice over the
first seconds of a generation, so the warm-up is cut off at the pause that
follows it, and the remaining clip (the reference text alone) is cloned with a
Qwen3-TTS Base checkpoint. Whisper word timestamps say where the reference text's
opening words are heard; the cut is the silence nearest that point, and the kept
clip is transcribed again to confirm it opens on the reference text, so the
stored transcript matches the stored audio. The result is a fixed voice identity, unlike a
description used at synthesis time, which samples a slightly different voice on
every call. See docs/voice-pack.md for the same method applied to the built-in voices.
"""
from __future__ import annotations

import threading
from typing import Any, Callable, List, Optional, Sequence, Tuple

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


# Design runs are serialized process-wide (each loads its own models).
_DESIGN_LOCK = threading.Lock()

# Whisper locates the reference; the cut is the silence nearest to it.
ALIGNMENT_WHISPER_MODEL = "base"
_MIN_PAUSE_S = 0.1
_MAX_CUT_DISTANCE_S = 0.6
_CUT_JITTER_S = 0.15
_MIN_REFERENCE_S = 4.0

# (text, start seconds) per transcribed word, for 16 kHz mono float32 audio.
Transcriber = Callable[[np.ndarray, str], Sequence[Tuple[str, float]]]


def _norm_chars(text: str) -> str:
    import unicodedata

    return "".join(ch for ch in unicodedata.normalize("NFKC", str(text)).lower() if ch.isalnum())


def find_pauses(audio: np.ndarray, sample_rate: int, *, min_pause_s: float = _MIN_PAUSE_S) -> List[int]:
    """Sample index at the centre of every silence of at least ``min_pause_s``."""
    a = np.asarray(audio, dtype=np.float32).reshape(-1)
    sr = int(sample_rate)
    frame, hop = int(0.02 * sr), int(0.01 * sr)
    if len(a) < frame * 10:
        return []
    rms = np.array([np.sqrt(np.mean(a[i:i + frame] ** 2)) for i in range(0, len(a) - frame, hop)])
    silent = rms < max(0.006, 0.12 * float(np.percentile(rms, 90)))
    centers, start = [], None
    for i, s in enumerate(np.append(silent, False)):
        if s and start is None:
            start = i
        elif not s and start is not None:
            if (i - start) * hop >= min_pause_s * sr:
                centers.append((start + i) // 2 * hop + frame // 2)
            start = None
    return centers


_HEAD_CHARS = 12
_MIN_HEAD_MATCH = 0.6


def _head_similarity(heard: str, expected: str) -> float:
    from difflib import SequenceMatcher

    return SequenceMatcher(None, heard, expected).ratio() if heard and expected else 0.0


def reference_start_s(words: Sequence[Tuple[str, float]], *, reference_text: str) -> Optional[float]:
    """When the reference text starts: the start of the transcribed word where the
    reference's opening characters are found (fuzzy, so a misheard word is
    tolerated; character-based, so scripts without spaces work). None when the
    opening is not heard clearly enough."""
    chars: List[str] = []
    starts: List[float] = []
    for word, start in words:
        for ch in _norm_chars(word):
            chars.append(ch)
            starts.append(float(start))
    head = _norm_chars(reference_text)[:_HEAD_CHARS]
    if not chars or not head:
        return None
    stream = "".join(chars)
    best, best_score = None, (0.0, 0)
    for i in range(len(stream)):
        window = stream[i:i + len(head)]
        # Similarity first; among equals, the window aligned on the opening wins
        # (a window shifted by one character can score the same).
        prefix = next((n for n, (a, b) in enumerate(zip(window, head)) if a != b), min(len(window), len(head)))
        score = (_head_similarity(window, head), prefix)
        if score > best_score:
            best, best_score = i, score
    if best is None or best_score[0] < _MIN_HEAD_MATCH:
        return None
    return starts[best]


def starts_with_reference(words: Sequence[Tuple[str, float]], *, reference_text: str) -> bool:
    """Whether the transcribed opening of a clip is the reference text's opening."""
    heard = "".join(_norm_chars(w) for w, _ in words)
    head = _norm_chars(reference_text)[:_HEAD_CHARS]
    return _head_similarity(heard[: len(head)], head) >= _MIN_HEAD_MATCH


def find_warmup_cut(audio: np.ndarray, sample_rate: int, *, reference_start: float) -> Optional[int]:
    """Sample index where the warm-up ends, given when the reference starts (seconds).

    The last silence shortly before the reference start (up to 0.6 s before, or
    0.15 s after for timestamp jitter) -- never a later one, which would cut the
    reference's first words. When the warm-up runs into the reference without a
    silence, the quietest instant in the 0.25 s before the start. None for an
    empty or too-short clip.
    """
    a = np.asarray(audio, dtype=np.float32).reshape(-1)
    sr = int(sample_rate)
    target = float(reference_start) * sr
    pauses = [c for c in find_pauses(a, sr) if target - _MAX_CUT_DISTANCE_S * sr <= c <= target + _CUT_JITTER_S * sr]
    if pauses:
        return max(pauses)
    frame, hop = int(0.02 * sr), int(0.005 * sr)
    lo, hi = max(0, int(target - 0.25 * sr)), min(len(a), int(target))
    if hi - lo < frame:
        return None
    energy = [float(np.mean(a[k:k + frame] ** 2)) for k in range(lo, hi - frame + 1, hop)]
    return lo + int(np.argmin(energy)) * hop + frame // 2


def whisper_transcriber(*, allow_downloads: bool, model: str = ALIGNMENT_WHISPER_MODEL) -> Transcriber:
    """faster-whisper word timestamps (``abstractvoice[stt]``); raises with the install command when missing."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "design_voice locates the reference with Whisper: install it with "
            'pip install "abstractvoice[stt]"'
        ) from exc
    whisper = WhisperModel(model, device="cpu", compute_type="int8", local_files_only=not allow_downloads)

    def transcribe(audio16k: np.ndarray, language: str) -> List[Tuple[str, float]]:
        segments, _info = whisper.transcribe(audio16k, language=language, word_timestamps=True)
        return [(w.word, float(w.start)) for seg in segments for w in (seg.words or [])]

    return transcribe


def _to_16k(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    a = np.asarray(audio, dtype=np.float32).reshape(-1)
    n = int(round(len(a) * 16000 / int(sample_rate)))
    return np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a).astype(np.float32)


def render_designed_reference(
    description: str,
    *,
    language: str = "en",
    text: Optional[str] = None,
    model_id: str = DESIGN_MODEL_ID,
    allow_downloads: bool = True,
    attempts: int = 3,
    settings: Any = None,
    transcriber: Optional[Transcriber] = None,
) -> Tuple[np.ndarray, int, str]:
    """Render ``(audio, sample_rate, reference_text)``: the settled voice reading ``text``.

    ``settings`` (a ``Qwen3TTSSettings``) selects the backend and precision.
    Raises ValueError for an empty description, an unsupported language, or a
    ``text`` too long for one generation or too short for a reference;
    RuntimeError when no attempt yields a clean cut.
    """
    from ..qwen3_tts.runtime import ONE_SHOT_MAX_CHARS

    if not str(description or "").strip():
        raise ValueError("a voice description is required")
    lang = str(language or "en").strip().lower()
    if lang not in WARMUPS:
        raise ValueError(f"design_voice supports the Qwen3-TTS languages {', '.join(WARMUPS)} (got {language!r})")
    reference_text = str(text).strip() if text and str(text).strip() else TEXTS[lang]
    warmup = WARMUPS[lang]
    full = f"{warmup} {reference_text}"
    if len(full) > ONE_SHOT_MAX_CHARS:
        raise ValueError(
            f"text is too long: with the {len(warmup)}-character warm-up it must fit one generation "
            f"({ONE_SHOT_MAX_CHARS} characters); use at most {ONE_SHOT_MAX_CHARS - len(warmup) - 1}"
        )
    with _DESIGN_LOCK:  # one design at a time: each loads a 1.7B model and Whisper
        return _render(
            str(description), lang, warmup, reference_text, full,
            model_id=model_id, allow_downloads=bool(allow_downloads), attempts=attempts,
            settings=settings, transcriber=transcriber, custom_text=bool(text and str(text).strip()),
        )


def _render(description, lang, warmup, reference_text, full, *, model_id, allow_downloads, attempts, settings,
            transcriber, custom_text):
    from ..adapters.tts_qwen3_tts import _qwen_language
    from ..qwen3_tts.runtime import Qwen3TTSRuntime, one_shot_max_new_tokens

    words_of = transcriber or whisper_transcriber(allow_downloads=allow_downloads)
    runtime = Qwen3TTSRuntime(model_id=model_id, allow_downloads=allow_downloads, settings=settings)
    too_short = 0
    try:
        for _ in range(max(1, int(attempts))):
            audio, sr = runtime.synthesize_voice_design(
                full,
                instruct=str(description),
                language=_qwen_language(lang),
                max_new_tokens=one_shot_max_new_tokens(full),
            )
            start = reference_start_s(words_of(_to_16k(audio, sr), lang), reference_text=reference_text)
            cut = None if start is None else find_warmup_cut(audio, sr, reference_start=start)
            if cut is None:
                continue
            ref = np.asarray(audio[cut:], dtype=np.float32).copy()
            if len(ref) / sr < _MIN_REFERENCE_S:
                too_short += 1
                continue
            # The kept clip must open on the reference text, not on the warm-up's tail.
            if not starts_with_reference(words_of(_to_16k(ref[: int(4 * sr)], sr), lang), reference_text=reference_text):
                continue
            n = int(0.01 * sr)
            ref[:n] *= np.linspace(0.0, 1.0, n, dtype=np.float32)  # 10 ms fade-in at the cut
            return ref, int(sr), reference_text
        if too_short and custom_text:
            raise ValueError(
                f"text is too short: the reference must last at least {_MIN_REFERENCE_S:.0f} s; use a longer text"
            )
        raise RuntimeError("could not separate the warm-up from the reference; try again or rephrase the description")
    finally:
        runtime.unload()
