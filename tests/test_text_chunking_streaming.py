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

    batches = split_complete_text_for_streaming(text, max_chars=180, first_max_chars=48)

    assert batches[0] == "CONFIRMED:"
    assert len(batches) < 20
    assert "this first phrase" in " ".join(batches)
