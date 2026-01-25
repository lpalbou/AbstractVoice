from abstractvoice.stop_phrase import is_stop_phrase


def test_stop_phrase_matches_inside_transcription():
    assert is_stop_phrase("stop.", ["stop"]) is True
    assert is_stop_phrase("stop please", ["stop"]) is True
    assert is_stop_phrase("okay stop", ["ok stop", "okay stop"]) is True
    assert is_stop_phrase("don't stop now", ["stop"]) is False
    assert is_stop_phrase("okey stop", ["ok stop", "okay stop"]) is True
    assert is_stop_phrase("oh stop", ["ok stop", "okay stop"]) is True
    assert is_stop_phrase("unrelated", ["stop"]) is False

