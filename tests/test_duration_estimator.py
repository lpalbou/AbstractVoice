from abstractvoice.tts.duration_estimator import count_speech_units, estimate_duration_s


def test_count_speech_units_words_for_en():
    assert count_speech_units("hello world", language="en") == 2
    assert count_speech_units("  hello   world  ", language="en") == 2


def test_count_speech_units_cjk_for_zh():
    assert count_speech_units("你好 世界", language="zh") == 4
    # Fallback when there are no CJK characters.
    assert count_speech_units("hello world", language="zh") == 2


def test_estimate_duration_clamps_min():
    assert estimate_duration_s("", language="en", min_s=0.2) == 0.2


def test_estimate_duration_clamps_max():
    d = estimate_duration_s("hello world", language="en", units_per_second=0.5, max_s=1.0)
    assert d == 1.0

