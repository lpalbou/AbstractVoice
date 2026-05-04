from abstractvoice.examples.cli_repl import strip_think_blocks
from abstractvoice.examples.llm_provider import LLMProvider, strip_think_blocks as provider_strip_think_blocks


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


def test_llm_provider_chat_extracts_and_strips_think_blocks(monkeypatch) -> None:
    captured = {}

    class Response:
        text = "fallback"

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [{"message": {"content": "<think>hidden</think>\nVisible reply."}}],
                "usage": {"completion_tokens": 3},
            }

    def fake_post(url, json, timeout):
        captured.update({"url": url, "json": json, "timeout": timeout})
        return Response()

    monkeypatch.setattr("abstractvoice.examples.llm_provider.requests.post", fake_post)

    provider = LLMProvider("dummy", "http://localhost:11434")
    result = provider.chat(
        model="local-model",
        messages=[{"role": "user", "content": "Hi"}],
        temperature=0.2,
        max_tokens=64,
    )

    assert result["text"] == "Visible reply."
    assert result["usage"] == {"completion_tokens": 3}
    assert captured["url"] == "http://localhost:11434/v1/chat/completions"
    assert captured["json"]["stream"] is False
    assert captured["json"]["messages"] == [{"role": "user", "content": "Hi"}]
    assert provider_strip_think_blocks("<think>x</think>ok") == "ok"
