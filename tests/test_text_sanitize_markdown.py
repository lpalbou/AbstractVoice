from __future__ import annotations

from abstractvoice.text_sanitize import sanitize_markdown_for_speech


def test_sanitize_removes_bold_single_word() -> None:
    assert sanitize_markdown_for_speech("Hello **world**!") == "Hello world!"


def test_sanitize_removes_bold_multiple_words() -> None:
    assert sanitize_markdown_for_speech("**sentence with multiple words**") == "sentence with multiple words"


def test_sanitize_removes_italic_single_word() -> None:
    assert sanitize_markdown_for_speech("Use *italics* here.") == "Use italics here."


def test_sanitize_removes_italic_multiple_words() -> None:
    assert sanitize_markdown_for_speech("Use *a sentence with words* here.") == "Use a sentence with words here."


def test_sanitize_strips_markdown_headers_keep_text() -> None:
    # Each heading is its own sentence; "###" alone is an empty heading (nothing to say).
    txt = "# Title\n## Subtitle\n##### Small header\nNot a header\n###"
    assert sanitize_markdown_for_speech(txt) == "Title.\nSubtitle.\nSmall header.\nNot a header"


def test_sanitize_drops_a_bullet_marker_and_keeps_a_lone_star() -> None:
    assert sanitize_markdown_for_speech("* item") == "item"
    assert sanitize_markdown_for_speech("2 * 3 * 4") == "2 * 3 * 4"

