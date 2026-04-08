from abstractvoice.tts.text_chunking import TextStreamChunker, TextStreamChunkingConfig, split_text_batches


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

