import pytest

from abstractvoice.tts.text_chunking import (
    TextStreamChunker,
    TextStreamChunkingConfig,
    split_complete_text_for_streaming,
    split_text_batches,
)


def test_split_text_batches_prefers_sentence_boundaries():
    s = "Hello world. This is a test! Another sentence?"
    batches = split_text_batches(s, max_chars=20)
    assert isinstance(batches, list)
    assert all(isinstance(b, str) for b in batches)
    assert all(len(b) <= 20 for b in batches)
    # Should not drop content.
    joined = " ".join(batches).replace("  ", " ").strip()
    assert "Hello world." in joined


def test_text_stream_chunker_emits_on_sentence_terminator():
    ch = TextStreamChunker(config=TextStreamChunkingConfig(max_chars=240, min_chars=1))
    out = []
    out += ch.push("Hello world")
    assert out == []
    out += ch.push(". ")
    assert out == ["Hello world."]


def test_text_stream_chunker_hard_cuts_when_no_boundaries():
    ch = TextStreamChunker(config=TextStreamChunkingConfig(max_chars=10, min_chars=1))
    out = ch.push("abcdefghijklmno")
    assert out
    assert all(len(x) <= 10 for x in out)


def test_split_complete_text_for_streaming_keeps_fast_first_segment_but_batches_rest():
    text = (
        "CONFIRMED: this first phrase should be emitted quickly. "
        "The rest of this complete response should be merged into efficient sentence-sized batches, "
        "instead of being split at every comma, colon, or other soft pause. "
    ) * 5

    batches = split_complete_text_for_streaming(text, max_chars=180)

    assert batches[0] == "CONFIRMED: this first phrase should be emitted quickly."
    assert len(batches) < 20
    assert " ".join(batches) == " ".join(text.split())


@pytest.mark.parametrize("text", ["测试这是连续的中文句子。" * 80, "日本語の長い文章です。" * 80, "a" * 1000, "a" * 1000 + ". Next sentence."])
@pytest.mark.parametrize("splitter", [split_text_batches, split_complete_text_for_streaming])
def test_multilingual_batches_enforce_hard_limit_without_losing_text(text, splitter):
    batches = splitter(text, max_chars=200)
    assert all(0 < len(chunk) <= 200 for chunk in batches)
    assert "".join(batches).replace(" ", "") == text.replace(" ", "")


def test_incremental_chunker_respects_cap_before_a_late_sentence_boundary():
    chunker = TextStreamChunker(config=TextStreamChunkingConfig(max_chars=200))
    text = "a" * 1000 + ". Next sentence."
    chunks = chunker.push(text) + chunker.flush()
    assert all(len(chunk) <= 200 for chunk in chunks)
    assert "".join(chunks).replace(" ", "") == text.replace(" ", "")


# Round 17: measured on the public gateway 2026-10-17, the first segment "Ah oui," (1.3 s of
# audio) was followed by 1.7-3.3 s of silence, and two words alone came out in the wrong
# language ("Ah oui," heard as English "Are we?").
_REPLIES = [
    "Ah oui, je vois très bien ce que tu veux dire. Le Mont Everest est la plus haute montagne du monde. "
    "Il se trouve à la frontière entre le Népal et la Chine, et il a été gravi pour la première fois en 1953.",
    "Oui. Voici le résumé de la réunion : le budget est reporté à jeudi, Marie enverra les diapositives ce soir, "
    "et nous reparlerons du recrutement la semaine prochaine. Est-ce que tu veux que je prépare un brouillon ?",
    "Sure, here is the short version: the meeting moved to Thursday at ten, the budget review is postponed, "
    "and Marie will send the updated slides tonight. Do you want me to draft an email?",
    "OK. " + "The gateway reads this reply aloud while the next sentence is synthesised. " * 40,
    "Oui. Non. Peut-être. Bien sûr. D'accord. Merci. " * 6,
]


@pytest.mark.parametrize("text", _REPLIES)
def test_every_segment_is_whole_sentences_and_none_is_a_fragment(text):
    segments = split_complete_text_for_streaming(text)
    assert " ".join(segments) == " ".join(text.split())
    assert all(x[-1] in ".!?" for x in segments), [x for x in segments if x[-1] not in ".!?"]
    assert len(segments[0]) >= 30, segments[0]
    assert all(len(x) >= 20 for x in segments), [x for x in segments if len(x) < 20]
    assert all(len(x) <= 240 for x in segments)


def test_the_first_segment_is_the_first_sentence_never_a_clause():
    text = _REPLIES[0]
    assert split_complete_text_for_streaming(text)[0] == "Ah oui, je vois très bien ce que tu veux dire."
    # "Oui." alone is too short to be spoken alone: it goes with the next sentence, whole.
    assert split_complete_text_for_streaming(_REPLIES[1])[0].startswith("Oui. Voici le résumé de la réunion :")
    assert split_complete_text_for_streaming(_REPLIES[1])[0].endswith("la semaine prochaine.")


def test_only_a_sentence_longer_than_the_cap_is_cut():
    long_sentence = "Le budget est reporté, " * 14 + "et nous en reparlerons."
    segments = split_complete_text_for_streaming("Bonjour à toutes et à tous. " + long_sentence, max_chars=240)
    assert segments[0] == "Bonjour à toutes et à tous."
    assert all(len(x) <= 240 for x in segments) and len(segments) >= 3


@pytest.mark.parametrize("text", ["Ah oui, je vois très bien ce que tu veux dire.", "Oui. Bien sûr, avec plaisir !",
                                  "Ah oui, je vois très bien ce que tu veux dire, et voici ce que je propose pour la suite."])
def test_a_one_sentence_reply_is_one_segment(text):
    assert split_complete_text_for_streaming(text) == [text]


@pytest.mark.parametrize("text", _REPLIES)
@pytest.mark.parametrize("synth_ratio,overhead_s", [(0.36, 0.5), (0.4, 0.6)])
def test_every_segment_is_made_before_the_previous_audio_ends(text, synth_ratio, overhead_s):
    """Simulated with the public gateway's measured pace (15 chars/s of speech, synthesis at
    0.36x real time + 0.5 s per call) and the cautious pace the splitter assumes."""
    made_at = played_at = 0.0
    gaps = []
    for i, seg in enumerate(split_complete_text_for_streaming(text)):
        audio = len(seg) / 15.0
        made_at += overhead_s + synth_ratio * audio
        if i and made_at > played_at + 0.05:
            gaps.append(round(made_at - played_at, 2))
        played_at = max(played_at, made_at) + audio
    assert gaps == []


def test_the_old_earliest_pause_split_had_the_measured_gap():
    """The check above fails on the former policy (first segment = text up to the earliest pause,
    then 240-character batches): the gap measured on the gateway."""
    text = _REPLIES[0]
    segments = ["Ah oui,"] + split_text_batches(text[len("Ah oui,"):].strip(), max_chars=240)
    made_at = played_at = 0.0
    gaps = []
    for i, seg in enumerate(segments):
        audio = max(1.3, len(seg) / 15.0)
        made_at += 0.5 + 0.36 * audio
        if i and made_at > played_at + 0.05:
            gaps.append(made_at - played_at)
        played_at = max(played_at, made_at) + audio
    assert gaps and gaps[0] > 1.5
