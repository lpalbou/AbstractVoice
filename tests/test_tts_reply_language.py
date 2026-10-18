"""Round 17: one language for a whole streamed reply, identified once on the whole text.

On Supertonic 3 with the default `en`, whole-sentence pieces still came out in the wrong
language (Whisper, 2026-10-18: "Le Mont Everest est la plus haute montagne du monde." heard
as German, "Le gateway est bien configuré." as English, 3-6 of 22 pieces); with the reply's
language identified once and held, 0 of 22."""
from __future__ import annotations

import numpy as np
import pytest

from abstractvoice.tts.language_id import identify_language
from abstractvoice.vm.tts_mixin import TtsMixin

SUPERTONIC = ["en", "ko", "ja", "ar", "bg", "cs", "da", "de", "el", "es", "et", "fi", "fr", "hi", "hr", "hu", "id",
              "it", "lt", "lv", "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "tr", "uk", "vi"]

FRENCH = ("Oui. Voici le résumé de la réunion : le budget est reporté à jeudi, Marie enverra les diapositives "
          "ce soir, et nous reparlerons du recrutement la semaine prochaine. Est-ce que tu veux un brouillon ?")


@pytest.mark.parametrize("text,lang", [
    (FRENCH, "fr"),
    ("Sure, here is the short version: the meeting moved to Thursday at ten.", "en"),
    ("Der Termin ist auf Donnerstag verschoben, und Marie schickt heute Abend die Folien.", "de"),
    ("La reunión se movió al jueves a las diez y Marie enviará las diapositivas esta noche.", "es"),
    ("会議は木曜日の十時に移動しました。資料は今夜送ります。", "ja"),
])
def test_a_whole_reply_is_identified(text, lang):
    assert identify_language(text, SUPERTONIC) == lang


@pytest.mark.parametrize("text", ["Ah oui,", "Oui.", "OK, super !"])
def test_a_fragment_is_not_guessed(text):
    assert identify_language(text, SUPERTONIC) is None


class _PerCallAdapter:
    engine_id = "fake"
    language_per_call = True

    def __init__(self):
        self._language = "en"
        self.calls: list[tuple[str, str]] = []

    def is_available(self):
        return True

    def get_max_chars(self):
        return 300

    def get_supported_languages(self):
        return list(SUPERTONIC)

    def set_language(self, language):
        self._language = language
        return True

    def synthesize_to_audio_chunks(self, text):
        self.calls.append((self._language, text))
        yield np.full(240, 0.1, dtype=np.float32), 24000


class _VM(TtsMixin):
    def __init__(self, adapter):
        self.tts_adapter = adapter
        self.language = "en"
        self.speed = 1.0
        self.metrics = None

    def _set_last_tts_metrics(self, metrics):
        self.metrics = metrics


def test_every_segment_of_a_french_reply_is_spoken_in_french_then_the_engine_is_restored():
    adapter = _PerCallAdapter()
    vm = _VM(adapter)
    list(vm.speak_to_audio_chunks(FRENCH))
    assert len(adapter.calls) >= 2
    assert {lang for lang, _ in adapter.calls} == {"fr"}, adapter.calls
    assert adapter._language == "en"
    assert vm.metrics["language"] == "fr" and vm.metrics["language_source"] == "reply"


def test_the_callers_language_wins():
    adapter = _PerCallAdapter()
    list(_VM(adapter).speak_to_audio_chunks(FRENCH, language="it"))
    assert {lang for lang, _ in adapter.calls} == {"it"}


def test_an_engine_whose_language_is_its_voice_model_is_left_alone():
    adapter = _PerCallAdapter()
    adapter.language_per_call = False
    vm = _VM(adapter)
    list(vm.speak_to_audio_chunks(FRENCH))
    assert {lang for lang, _ in adapter.calls} == {"en"}
    assert vm.metrics["language_source"] == "configured"
