"""ADR 0009: streamed speech marks its text segments, located exactly in the caller's text. Weight-free.

A client that lost a stream re-requests the text from ``text_end`` of the last segment it
received whole, and drops the audio of the partial one: whatever the engine samples on the
retry, no word is repeated or missed. These tests prove the contract the client relies on:

- every audio item / line belongs to the segment opened before it and closed after it;
- the spans are ordered, cover every spoken character, and refer to the ORIGINAL text
  (Markdown markers removed and whitespace collapsed before synthesis are mapped back);
- the remainder after any segment speaks exactly the text not yet spoken;
- the Qwen3-TTS clone engine marks its generations in delivery order, through the
  speaker check (a regenerated segment is one segment) and its outbox pacing.
"""
from __future__ import annotations

import re
import threading

import numpy as np
import pytest

from abstractvoice.cloning.engine_qwen3_tts import Qwen3TTSVoiceCloningEngine
from abstractvoice.integrations.abstractcore_plugin import _VoiceCapability
from abstractvoice.qwen3_tts.runtime import Qwen3TTSSettings
from abstractvoice.text_sanitize import sanitize_markdown_for_speech
from abstractvoice.tts.segments import SegmentEnd, SegmentLocator, SegmentStart, audio_only
from abstractvoice.tts.text_chunking import split_complete_text_for_streaming, split_text_batches
from abstractvoice.vm.tts_mixin import TtsMixin

SR = 24000

FRENCH = (
    "Bonjour Laurent ! Il est 14 h 30, et la température extérieure atteint 21,5 °C. "
    "« C'est l'été indien », dit-elle ; l'œuvre coûte 1 250,99 € (TVA à 20 %). "
    "Voilà 👍🏽 — on part à 9h, d'accord ? Les élèves ont lu 3 chapitres… puis ils ont dormi. "
    "Où est passé l'été ?   Déjà fini.\n\nÀ demain 👨‍👩‍👧 !"
)
ENGLISH = (
    "It's 3.14 o'clock: version 2.0 shipped on 2026-11-02, costing $1,299.99 (or 15% off). "
    "Really?! Yes - twice... Then, at 10 p.m., the 3rd test passed 🎉. "
    "Next: emails, e.g. a@b.co; URLs like https://x.y/z?q=1&r=2 stay as they are. "
    + " ".join(f"Sentence number {i} is here to make the reply long enough for several segments." for i in range(12))
)


def _spoken(text: str) -> str:
    """What an engine speaks of ``text``: Markdown markers removed, whitespace collapsed."""
    return re.sub(r"\s+", " ", sanitize_markdown_for_speech(text)).strip()


# ----------------------------------------------------------------- locator


@pytest.mark.parametrize("text", [FRENCH, ENGLISH], ids=["french-emoji", "english-numbers"])
@pytest.mark.parametrize("split", [split_text_batches, split_complete_text_for_streaming], ids=["batches", "streaming"])
def test_segment_spans_are_exact_slices_of_the_original_text(text, split):
    segments = split(_spoken(text), max_chars=80)
    assert len(segments) >= 3
    locator = SegmentLocator(text, sanitize=True)
    spans = [locator.locate(seg) for seg in segments]
    previous = 0
    for seg, (start, end) in zip(segments, spans):
        assert previous <= start < end <= len(text)
        assert re.sub(r"\s+", " ", text[start:end]) == seg  # plain text: the span IS the segment
        assert text[previous:start].strip() == ""  # nothing spoken is skipped between segments
        previous = end
    assert text[previous:].strip() == ""
    # Resuming after any segment speaks exactly the segments not yet spoken.
    for k, (_, end) in enumerate(spans):
        assert _spoken(text[end:]) == " ".join(segments[k + 1:])


def test_markdown_markers_map_back_to_the_original_text():
    text = "## Plan du jour\n\n**Première étape.** Lire le *rapport*. Puis **écrire la suite**, enfin dormir."
    spoken = _spoken(text)
    # The heading is its own sentence: its full stop is inserted, and stands for its last character.
    assert spoken == "Plan du jour. Première étape. Lire le rapport. Puis écrire la suite, enfin dormir."
    locator = SegmentLocator(text, sanitize=True)
    first = locator.locate("Plan du jour. Première étape.")
    assert text[first[0]:first[1]] == "Plan du jour\n\n**Première étape.**"  # the markers around it belong to it
    second = locator.locate("Lire le rapport.")
    assert text[second[0]:second[1]] == "Lire le *rapport*."
    third = locator.locate("Puis écrire la suite, enfin dormir.")
    assert text[third[0]:third[1]] == "Puis **écrire la suite**, enfin dormir."
    assert _spoken(text[first[1]:]) == "Lire le rapport. Puis écrire la suite, enfin dormir."


def test_a_segment_that_is_not_the_next_part_of_the_text_fails_loudly():
    locator = SegmentLocator("One. Two. Three.", sanitize=True)
    assert locator.locate("One.") == (0, 4)
    with pytest.raises(RuntimeError, match="not the next part"):
        locator.locate("One.")  # already located: offsets never go backwards
    with pytest.raises(RuntimeError, match="not the next part"):
        SegmentLocator("One. Two.", sanitize=True).locate("Uno.")


def test_audio_only_drops_markers_and_closes_the_source():
    closed = []

    def source():
        try:
            yield SegmentStart(0, "a")
            yield np.zeros(3), SR
            yield SegmentEnd(0)
            yield SegmentStart(1, "b")
            yield np.ones(3), SR
        finally:
            closed.append(True)

    stream = audio_only(source())
    assert next(stream)[0].tolist() == [0, 0, 0]
    stream.close()
    assert closed == [True]


# ----------------------------------------------------------------- base TTS (voice manager)


class _Adapter:
    engine_id = "fake"

    def __init__(self, *, native=False, seconds_per_char=0.07, max_chars=240):
        self.native_frame_streaming = native
        self.seconds_per_char = seconds_per_char
        self.max_chars = max_chars
        self.calls: list[str] = []

    def is_available(self):
        return True

    def get_max_chars(self):
        return self.max_chars

    def synthesize_to_audio_chunks(self, text):
        # The audio's samples encode which call made it; a native engine yields frames of one generation.
        self.calls.append(text)
        n = max(1, int(len(text) * self.seconds_per_char * SR))
        value = 0.01 * len(self.calls)
        step = SR // 2 if self.native_frame_streaming else n
        for start in range(0, n, step):
            yield np.full(min(step, n - start), value, dtype=np.float32), SR


class _VM(TtsMixin):
    def __init__(self, adapter):
        self.tts_adapter = adapter
        self.language = "en"
        self.speed = 1.0
        self.metrics = None

    def _set_last_tts_metrics(self, metrics):
        self.metrics = metrics


def _segments(events):
    """[(start_marker, [audio...], ended)] in order; asserts the marker order is well formed."""
    out = []
    for item in events:
        if isinstance(item, SegmentStart):
            assert not out or out[-1][2], "a segment starts before the previous one ended"
            assert item.index == len(out)
            out.append((item, [], False))
        elif isinstance(item, SegmentEnd):
            start, audio, ended = out[-1]
            assert item.index == start.index and not ended
            out[-1] = (start, audio, True)
        else:
            assert out and not out[-1][2], "audio outside a segment"
            out[-1][1].append(item)
    return out


@pytest.mark.parametrize("text", [FRENCH, ENGLISH], ids=["french-emoji", "english-numbers"])
def test_base_tts_marks_each_synthesised_segment_with_its_span(text):
    adapter = _Adapter(max_chars=120)
    vm = _VM(adapter)
    segs = _segments(vm.speak_to_audio_segments(text))
    assert len(segs) == len(adapter.calls) >= 3
    for k, (start, audio, ended) in enumerate(segs):
        assert ended and start.text == adapter.calls[k]
        assert re.sub(r"\s+", " ", text[start.text_start:start.text_end]) == start.text
        assert audio and all(float(a[0]) == pytest.approx(0.01 * (k + 1)) for a, _ in audio)  # only that call's audio
    assert vm.metrics["segments"] == len(segs)


def test_base_tts_resume_after_a_break_speaks_exactly_the_rest():
    text = ENGLISH
    first = _segments(_VM(_Adapter(max_chars=120)).speak_to_audio_segments(text))
    for k in range(len(first) - 1):
        end = first[k][0].text_end
        adapter = _Adapter(max_chars=120)
        retry = _segments(_VM(adapter).speak_to_audio_segments(text[end:]))
        assert " ".join(s.text for s, _, _ in retry) == " ".join(s.text for s, _, _ in first[k + 1:])
        # The retry's offsets refer to the retry's own text.
        assert all(text[end:][s.text_start:s.text_end].strip() for s, _, _ in retry)


def test_a_native_streaming_engine_is_one_segment_over_the_whole_text():
    text = "  First sentence.\n\nSecond one, with 2 numbers: 3 and 4.  "
    adapter = _Adapter(native=True)
    segs = _segments(_VM(adapter).speak_to_audio_segments(text))
    assert len(segs) == 1 and len(segs[0][1]) > 1  # frames of one generation
    start, _, ended = segs[0]
    assert ended and (start.index, start.text_start, start.text_end) == (0, 2, len(text) - 2)


def test_a_cancelled_segment_never_ends():
    cancel = threading.Event()
    vm = _VM(_Adapter(max_chars=60))
    seen = []
    for item in vm.speak_to_audio_segments(ENGLISH, cancel_event=cancel):
        seen.append(item)
        if isinstance(item, tuple) and sum(isinstance(i, SegmentEnd) for i in seen) == 1:
            cancel.set()  # the second segment's first audio is in hand
    segs = _segments(seen)
    assert [ended for _, _, ended in segs] == [True, False]


# ----------------------------------------------------------------- the stream lines (plugin)


class _Owner:
    def __init__(self, vm):
        self.config = {"voice_manager_instance": vm}


def _stream(vm, text):
    return list(_VoiceCapability(_Owner(vm)).tts_stream(text, cancel_event=threading.Event()))


def test_stream_lines_carry_the_segment_and_its_span_and_end_marks_follow_the_last_slice():
    # 0.2 s per char: a segment of ~100 chars is ~20 s of audio, sliced into 2 s lines.
    vm = _VM(_Adapter(max_chars=120, seconds_per_char=0.2))
    lines = _stream(vm, FRENCH)
    assert lines[-1]["type"] == "done" and lines[-1]["segments"] >= 3
    body = lines[:-1]
    assert {line["type"] for line in body} == {"audio", "segment_end"}
    current, ends = 0, []
    for line in body:
        if line["type"] == "audio":
            assert line["segment"] == current
        else:
            assert line["segment"] == current
            ends.append(line)
            current += 1
    assert len(ends) == lines[-1]["segments"]
    for end in ends:
        audio = [line for line in body if line["type"] == "audio" and line["segment"] == end["segment"]]
        assert len(audio) > 1  # sliced
        assert {(a["text_start"], a["text_end"]) for a in audio} == {(end["text_start"], end["text_end"])}
        assert re.sub(r"\s+", " ", FRENCH[end["text_start"]:end["text_end"]]) == vm.tts_adapter.calls[end["segment"]]
    assert [line["sequence"] for line in body if line["type"] == "audio"] == list(
        range(sum(line["type"] == "audio" for line in body))
    )
    # Ordering, segment by segment: exactly one segment_end, after the segment's last audio line and before the
    # next segment's first (a client that trusts segment_end must have heard the whole segment by then).
    count = lines[-1]["segments"]
    for k in range(count):
        audio_at = [i for i, line in enumerate(body) if line["type"] == "audio" and line["segment"] == k]
        end_at = [i for i, line in enumerate(body) if line["type"] == "segment_end" and line["segment"] == k]
        assert len(end_at) == 1, (k, end_at)
        assert audio_at and end_at[0] > max(audio_at)
        if k + 1 < count:
            next_audio = [i for i, line in enumerate(body) if line["type"] == "audio" and line["segment"] == k + 1]
            assert end_at[0] < min(next_audio)
    assert sum(line["type"] == "segment_end" for line in body) == count


def test_the_client_retry_rule_loses_and_repeats_nothing():
    """Break after every line in turn; resume from the last segment_end: the finished segments plus the retry's
    are the whole text (the boundary part of the hybrid rule; the seconds skip inside the unfinished segment is
    the client's)."""
    text = FRENCH + " " + ENGLISH
    lines = _stream(_VM(_Adapter(max_chars=120)), text)
    for cut in range(len(lines) - 1):
        received = lines[: cut + 1]
        ends = [line for line in received if line["type"] == "segment_end"]
        resume = ends[-1]["text_end"] if ends else 0
        heard = [text[e["text_start"]:e["text_end"]] for e in ends]  # the partial segment's audio is dropped
        retry_adapter = _Adapter(max_chars=120)
        retry = _stream(_VM(retry_adapter), text[resume:])
        rest = [text[resume:][e["text_start"]:e["text_end"]] for e in retry if e["type"] == "segment_end"]
        assert _spoken(" ".join(heard + rest)) == _spoken(text)


# ----------------------------------------------------------------- Qwen3-TTS clone engine


def _tag(segment, attempt, chunk):
    return 0.1 * (segment + 1) + 0.01 * attempt + 0.0001 * chunk


def _untag(value):
    value = float(value)
    segment = int(value * 10 + 1e-6) - 1
    rest = value - 0.1 * (segment + 1)
    attempt = int(rest * 100 + 1e-6)
    return segment, attempt


class _CloneRT:
    """Native (MLX) or whole-segment (torch) fake; ``voice(segment, attempt)`` is the speaker similarity."""

    def __init__(self, voice, *, native, seconds=6.0):
        self.settings = Qwen3TTSSettings()
        self.supports_native_streaming = native
        self.voice, self.seconds = voice, seconds
        self.texts: list[str] = []
        self.attempts: dict[str, int] = {}

    def build_clone_prompt(self, *, ref_audio, ref_text, x_vector_only):
        return "prompt"

    def _chunks(self, text):
        if text not in self.texts:
            self.texts.append(text)
        attempt = self.attempts.get(text, 0)
        self.attempts[text] = attempt + 1
        seg, n, step = self.texts.index(text), int(self.seconds * SR), int(0.96 * SR)
        for k, start in enumerate(range(0, n, step)):
            yield np.full(min(step, n - start), _tag(seg, attempt, k), dtype=np.float32), SR

    def stream_clone(self, text, *, clone_prompt, language=None, **kw):
        return self._chunks(text)

    def synthesize_clone(self, text, *, clone_prompt, language=None, **kw):
        return np.concatenate([a for a, _ in self._chunks(text)]), SR

    def speaker_similarity(self, audio, *, clone_prompt, sample_rate=SR):
        return min(float(self.voice(*_untag(v))) for v in np.unique(np.asarray(audio).reshape(-1)))


# Three ~800-char paragraphs: three generations on the native path, many at the 120-char torch bound.
CLONE_TEXT = " ".join(
    " ".join(f"Paragraphe {p}, phrase {i} : l'élève compte jusqu'à {i * 7} 👍." for i in range(16))
    for p in range(3)
)


@pytest.mark.parametrize("native", [True, False], ids=["mlx-native-stream", "torch-segments"])
def test_clone_engine_marks_generations_in_delivery_order_through_a_regeneration(native):
    # Segment 1's first take comes out in a stock voice and is regenerated (backlog 056).
    rt = _CloneRT(lambda seg, att: 0.90 if (seg, att) == (1, 0) else 0.98, native=native)
    engine = Qwen3TTSVoiceCloningEngine(allow_downloads=False)
    engine._runtime = rt
    events = list(engine.infer_to_audio_segments(
        text=CLONE_TEXT, reference_paths=["ref.wav"], reference_text="t", max_chars=120,
    ))
    segs = _segments(events)
    assert len(segs) == len(rt.texts) >= 3 and all(ended for _, _, ended in segs)
    for k, (start, audio, _) in enumerate(segs):
        assert start.text == rt.texts[k]
        takes = {_untag(a[0]) for a, _ in audio}
        assert takes == {(k, 1 if k == 1 else 0)}  # only the accepted take, all of it, inside its segment
        assert sum(a.size for a, _ in audio) == int(rt.seconds * SR)
    assert [a for a in audio_only(iter(events))] == [e for e in events if isinstance(e, tuple)]


@pytest.mark.parametrize("native", [True, False], ids=["mlx-native-stream", "torch-segments"])
def test_clone_engine_marks_generations_with_the_speaker_check_off(native):
    rt = _CloneRT(lambda seg, att: 0.50, native=native)  # would fail the check: proves it is off
    engine = Qwen3TTSVoiceCloningEngine(allow_downloads=False, speaker_check=False)
    engine._runtime = rt
    events = []
    for item in engine.infer_to_audio_segments(
        text=CLONE_TEXT, reference_paths=["ref.wav"], reference_text="t", max_chars=120,
    ):
        events.append(item)
        if isinstance(item, tuple):
            # Delivered as generated: a segment's audio is never held behind its end marker.
            assert _untag(item[0][0])[0] == len(rt.texts) - 1
    segs = _segments(events)
    assert len(segs) == len(rt.texts) >= 3 and all(ended for _, _, ended in segs)
    for k, (start, audio, _) in enumerate(segs):
        assert start.text == rt.texts[k] and {_untag(a[0]) for a, _ in audio} == {(k, 0)}


def test_cloned_stream_lines_locate_each_generation_in_the_request_text():
    rt = _CloneRT(lambda seg, att: 0.98, native=False, seconds=3.0)
    engine = Qwen3TTSVoiceCloningEngine(allow_downloads=False)
    engine._runtime = rt

    class _Cloner:
        def speak_to_audio_segments(self, text, *, voice_id, speed=None, max_chars=120, language=None):
            return engine.infer_to_audio_segments(
                text=text, reference_paths=["ref.wav"], reference_text="t", max_chars=max_chars,
            )

    class _CloneVM(_VM):
        def _get_voice_cloner(self):
            return _Cloner()

        def _effective_clone_speed(self, **kw):
            return 1.0, "qwen3-tts"

        def get_cloned_voice(self, voice_id):
            return {"voice_id": voice_id, "engine": "qwen3-tts"}

    text = "**Bonjour !**  " + CLONE_TEXT
    lines = list(
        _VoiceCapability(_Owner(_CloneVM(_Adapter()))).tts_stream(text, voice="sisko", cancel_event=threading.Event())
    )
    ends = [line for line in lines if line["type"] == "segment_end"]
    assert len(ends) == len(rt.texts) == lines[-1]["segments"] >= 3
    for end, seg_text in zip(ends, rt.texts):
        assert _spoken(text[end["text_start"]:end["text_end"]]) == seg_text
    assert text[: ends[0]["text_end"]].startswith("**Bonjour !**")
    assert ends[-1]["text_end"] == len(text)
