from abstractvoice.examples.cli_repl import strip_think_blocks


def test_strip_think_blocks_noop_when_missing() -> None:
    s = "Hello world.\n\nNo tags here."
    assert strip_think_blocks(s) == s.strip()


def test_strip_think_blocks_removes_single_block() -> None:
    s = "<think>secret\nreasoning</think>\n\nFinal answer."
    assert strip_think_blocks(s) == "Final answer."


def test_strip_think_blocks_removes_multiple_blocks() -> None:
    s = "A\n<think>one</think>\nB\n<think>two</think>\nC"
    assert strip_think_blocks(s) == "A\nB\nC"


def test_strip_think_blocks_case_insensitive_and_whitespace() -> None:
    s = "X\n<THINK>hidden</THINK>\nY"
    assert strip_think_blocks(s) == "X\nY"


def test_strip_think_blocks_unclosed_drops_tail() -> None:
    s = "Keep this.\n<think>do not show\nFinal answer maybe missing close"
    assert strip_think_blocks(s) == "Keep this."

