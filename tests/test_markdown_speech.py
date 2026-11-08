"""Markdown replies are read as their words, never their syntax or their code (weight-free).

2026-11-08: a read-aloud of a Code reply (a fenced Python call inside a list item, inline code,
headings, an arrow) sent the Markdown to a Qwen3-TTS clone almost verbatim. The engine read the
code as noise, slurred into a repeated syllable inside ``model="qwen/qwen3.8-27b")``, missed its
end of speech and played silence until its runaway cap; the voice came back two minutes later at
the next heading. ``REPLY`` is that reply (the private endpoint name replaced); the test below
pins the exact text of every segment the clone engine speaks of it.
"""
from __future__ import annotations

import re

import pytest

from abstractvoice import text_sanitize
from abstractvoice.text_sanitize import sanitize_markdown_for_speech, sanitize_markdown_with_index
from abstractvoice.tts.segments import SegmentLocator
from abstractvoice.tts.text_chunking import split_complete_text_for_streaming, split_text_batches

REPLY = """I've hit the real blocker, and I want to be straight with you rather than burn the last iteration on more digging.

## What I confirmed
- **The integration point is correct and I've verified the exact call:**
  ```python
  from abstractcore import create_llm
  llm = create_llm("endpoint:compat-example-gpu", model="qwen/qwen3.8-27b")
  resp = llm.generate(prompt, max_output_tokens=...)
  ```
- **The profile mechanism is clear:** `endpoint:compat-example-gpu` is a virtual provider that resolves (via `ProviderEndpointProfileStore` → Core's `provider_profiles`) to a concrete `base_url` + `api_key` for an OpenAI-compatible server.

## The blocker (root cause of the earlier "Circuit breaker open")
The profile **is not resolvable inside my sandbox**:
- The Core config that holds it lives at `~/.abstractcore/config/abstractcore.json`, which my sandbox **cannot read** (`PermissionError: Operation not permitted`).
- It's not registered in any local store I can reach, so `create_llm` raises `UnknownProviderError: endpoint:compat-example-gpu`.

So I can't make a live LLM call from here — not because the code is wrong, but because the endpoint's credentials/profile aren't visible to this environment.

## To unblock, I need one of these
1. **Run the project in the gateway/runtime environment** where the profile already resolves (the place that normally serves `qwen/qwen3.8-27b`), or
2. **Give me the `base_url` + `api_key`** for the example endpoint and I'll register the profile locally and test live, or
3. **Point me at a readable config file** that contains the example profile.

## The design I'll build (ready to go the moment the endpoint resolves)
- **Backend (Python/abstractcore):** an agent brain — each agent has a role + workstation purpose, a `think()` that calls the qwen model to decide *work vs. walk vs. talk*, and a `dialogue()` that exchanges turns with a nearby agent. A small FastAPI/WS server streams agent state + dialogue to the browser.
- **Frontend (three.js + Vite):** a 3D scene with N agents, each with a labeled workstation; agents wander, and when two come within range they stop and a dialogue bubble streams their conversation; otherwise they "work" at their station (progress indicator).
- **Collaboration:** agents share a lightweight blackboard (task queue / shared state) so their workstation roles feed each other, with the LLM deciding when to hand off.

**My recommendation:** option 1 or 2. If you can tell me the example `base_url` + key (or confirm I should run it under the gateway runtime), I'll wire the whole thing up and verify a live agent dialogue end-to-end. Which way do you want to go?"""

# What the Qwen3-TTS clone engine speaks of REPLY on the MLX backend: one generation per
# segment of at most NATIVE_STREAM_MAX_CHARS (800), cut at sentence ends (engine_qwen3_tts).
REPLY_SEGMENTS = [
    "I've hit the real blocker, and I want to be straight with you rather than burn the last iteration on more "
    "digging. What I confirmed. The integration point is correct and I've verified the exact call: The profile "
    "mechanism is clear: endpoint:compat-example-gpu is a virtual provider that resolves (via "
    "ProviderEndpointProfileStore, Core's provider_profiles) to a concrete base_url + api_key for an "
    "OpenAI-compatible server. The blocker (root cause of the earlier \"Circuit breaker open\"). The profile is not "
    "resolvable inside my sandbox: The Core config that holds it lives at ~/.abstractcore/config/abstractcore.json, "
    "which my sandbox cannot read (PermissionError: Operation not permitted).",
    "It's not registered in any local store I can reach, so create_llm raises UnknownProviderError: "
    "endpoint:compat-example-gpu. So I can't make a live LLM call from here — not because the code is wrong, but "
    "because the endpoint's credentials/profile aren't visible to this environment. To unblock, I need one of "
    "these. 1: Run the project in the gateway/runtime environment where the profile already resolves (the place "
    "that normally serves qwen/qwen3.8-27b), or. 2: Give me the base_url + api_key for the example endpoint and "
    "I'll register the profile locally and test live, or. 3: Point me at a readable config file that contains the "
    "example profile. The design I'll build (ready to go the moment the endpoint resolves).",
    "Backend (Python/abstractcore): an agent brain — each agent has a role + workstation purpose, a think() that "
    "calls the qwen model to decide work vs. walk vs. talk, and a dialogue() that exchanges turns with a nearby "
    "agent. A small FastAPI/WS server streams agent state + dialogue to the browser. Frontend (three.js + Vite): a "
    "3D scene with N agents, each with a labeled workstation; agents wander, and when two come within range they "
    "stop and a dialogue bubble streams their conversation; otherwise they \"work\" at their station (progress "
    "indicator). Collaboration: agents share a lightweight blackboard (task queue / shared state) so their "
    "workstation roles feed each other, with the LLM deciding when to hand off. My recommendation: option 1 or 2.",
    "If you can tell me the example base_url + key (or confirm I should run it under the gateway runtime), I'll "
    "wire the whole thing up and verify a live agent dialogue end-to-end. Which way do you want to go?",
]


def _spoken(text: str) -> str:
    return re.sub(r"\s+", " ", sanitize_markdown_for_speech(text)).strip()


def _clone_segments(text: str) -> list[str]:
    return split_text_batches(sanitize_markdown_for_speech(text), max_chars=800)


# ------------------------------------------------------------------- the 2026-11-08 reply


def test_the_reply_is_spoken_as_these_exact_segments():
    assert _clone_segments(REPLY) == REPLY_SEGMENTS


def test_no_code_and_no_markdown_syntax_reaches_the_engine():
    spoken = _spoken(REPLY)
    for gone in ("```", "`", "**", "## ", "→", "create_llm(", "max_output_tokens", "import create_llm", "\n- "):
        assert gone not in spoken, gone
    # Inline code is read as its literal text; the numbered options keep their numbers.
    for kept in ("~/.abstractcore/config/abstractcore.json", "endpoint:compat-example-gpu", "1: Run", "2: Give", "3: Point"):
        assert kept in spoken, kept


def test_the_check_goes_red_when_the_code_fence_rule_is_removed(monkeypatch):
    # Mutant: no fenced-code rule. The fixture assertion must fail, and the code is spoken again.
    monkeypatch.setattr(text_sanitize, "_FENCE_OPEN", re.compile(r"(?!)"))
    assert _clone_segments(REPLY) != REPLY_SEGMENTS
    assert 'create_llm("endpoint:compat-example-gpu", model="qwen/qwen3.8-27b")' in _spoken(REPLY)


@pytest.mark.parametrize("split", [split_text_batches, split_complete_text_for_streaming], ids=["batches", "streaming"])
@pytest.mark.parametrize("max_chars", [800, 240, 120])
def test_every_segment_maps_back_to_the_reply_and_resumes_exactly(split, max_chars):
    segments = split(_spoken(REPLY), max_chars=max_chars)
    locator = SegmentLocator(REPLY, sanitize=True)
    previous = 0
    for k, seg in enumerate(segments):
        start, end = locator.locate(seg)
        assert previous <= start < end <= len(REPLY)
        assert _spoken(REPLY[previous:start]) == ""  # only syntax (a bullet, a code block) between segments
        # A client that lost the stream after segment k re-sends the reply from text_end: it hears the rest exactly.
        assert _spoken(REPLY[end:]) == " ".join(segments[k + 1:])
        previous = end
    assert _spoken(REPLY[previous:]) == ""


def test_the_index_never_decreases_and_kept_characters_are_the_source_characters():
    for text in (REPLY, "Plain text, spoken exactly as written", "| a | b |\n|---|---|\n| 1 | 2 |\n\n> - [x] **done** → next"):
        out, index = sanitize_markdown_with_index(text)
        assert len(out) == len(index)
        assert all(a <= b for a, b in zip(index, index[1:]))
        inserted = set(".:, \n")
        assert all(ch == text[i] or ch in inserted for ch, i in zip(out, index))


# -------------------------------------------------------------------------- the rules


@pytest.mark.parametrize(
    "markdown, spoken",
    [
        # Plain text is spoken exactly as written.
        ("Hello world", "Hello world"),
        ("Two lines\nof one paragraph.", "Two lines\nof one paragraph."),
        # Fenced code is not read: ``` and ~~~, at any indent, a longer closing fence, or unclosed.
        ("Run this:\n```bash\nrm -rf build\n```\nThen restart.", "Run this:\nThen restart."),
        ("Before\n~~~\ncode\n~~~\nAfter", "Before.\nAfter"),
        ("- item:\n    ```\n    x = 1\n    ````\n- next", "item:\nnext"),
        ("Here:\n```python\nprint(1)\nno closing fence", "Here:"),
        ("``` not a fence ` because of this backtick", "not a fence  because of this backtick"),
        # Inline code: literal, backticks dropped; nothing inside is rewritten; emphasis around it pairs up.
        ("Call `__init__` with `a*b*c`", "Call __init__ with a*b*c"),
        ("**Give me the `base_url` + `api_key`** now", "Give me the base_url + api_key now"),
        ("a stray ` backtick", "a stray  backtick"),
        # Headings are sentences; a block followed by another ends with a full stop unless it has punctuation.
        ("# Title\n\nBody.", "Title.\nBody."),
        ("## Done!\nNext", "Done!\nNext"),
        ("Setext\n======\nBody", "Setext.\nBody"),
        ("## Closed ##", "Closed"),
        ("#hashtag is not a heading", "#hashtag is not a heading"),
        # Lists: bullets dropped, numbers kept as "N:", task boxes dropped.
        ("- one\n- two\n- three", "one.\ntwo.\nthree"),
        ("1. first\n2) second", "1: first.\n2: second"),
        ("- [x] shipped\n- [ ] pending", "shipped.\npending"),
        # Quotes, rules, tables.
        ("> quoted *text*\n>\n> second", "quoted text.\nsecond"),
        ("above\n\n---\n\nbelow", "above.\nbelow"),
        ("| Name | Role |\n|:---|---:|\n| Ada | **lead** |", "Name, Role.\nAda, lead"),
        # Links, images, autolinks, escapes, strikethrough, underscore emphasis (never inside a word).
        ("See [the docs](https://x.y/a_(b) \"t\") and ![a chart](c.png)", "See the docs and a chart"),
        ("Mail <mailto:a@b.co> or <https://x.y>", "Mail mailto:a@b.co or https://x.y"),
        (r"\*not italic\* and ~~gone~~ text", "not italic and gone text"),
        ("snake_case_name stays, _this_ is emphasis", "snake_case_name stays, this is emphasis"),
        # Arrows are a pause; a lone star or a dash between spaces is prose.
        ("A → B ⇒ C -> D", "A, B, C, D"),
        ("2 * 3 - 1 = 5", "2 * 3 - 1 = 5"),
        ("unpaired *star and walk vs. talk* end", "unpaired star and walk vs. talk end"),
    ],
)
def test_markdown_rule(markdown, spoken):
    assert sanitize_markdown_for_speech(markdown) == spoken
