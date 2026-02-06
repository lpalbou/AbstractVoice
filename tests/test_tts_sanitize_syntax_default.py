from __future__ import annotations

import pytest

from abstractvoice.vm.tts_mixin import TtsMixin


class _DummyTts(TtsMixin):
    def __init__(self) -> None:
        self.speed = 1.0
        self.tts_engine = None
        self.tts_adapter = None


def test_speak_sanitizes_markdown_by_default() -> None:
    captured: dict[str, str] = {}

    class FakeEngine:
        def speak(self, text, _speed, _callback):
            captured["text"] = str(text)
            return True

    vm = _DummyTts()
    vm.tts_engine = FakeEngine()

    vm.speak("# Title **bold** *italics*")
    assert captured["text"] == "Title bold italics"


def test_speak_can_opt_out_of_sanitization() -> None:
    captured: dict[str, str] = {}

    class FakeEngine:
        def speak(self, text, _speed, _callback):
            captured["text"] = str(text)
            return True

    vm = _DummyTts()
    vm.tts_engine = FakeEngine()

    vm.speak("# Title **bold**", sanitize_syntax=False)
    assert captured["text"] == "# Title **bold**"


def test_speak_accepts_saninitze_syntax_alias() -> None:
    captured: dict[str, str] = {}

    class FakeEngine:
        def speak(self, text, _speed, _callback):
            captured["text"] = str(text)
            return True

    vm = _DummyTts()
    vm.tts_engine = FakeEngine()

    vm.speak("# Title **bold**", saninitze_syntax=False)
    assert captured["text"] == "# Title **bold**"


def test_speak_rejects_conflicting_sanitize_flags() -> None:
    class FakeEngine:
        def speak(self, _text, _speed, _callback):
            return True

    vm = _DummyTts()
    vm.tts_engine = FakeEngine()

    with pytest.raises(ValueError):
        vm.speak("hi", sanitize_syntax=False, saninitze_syntax=True)


def test_speak_to_bytes_sanitizes_markdown_by_default() -> None:
    captured: dict[str, str] = {}

    class FakeAdapter:
        def is_available(self) -> bool:
            return True

        def synthesize_to_bytes(self, text, *, format: str = "wav") -> bytes:
            captured["text"] = str(text)
            assert format == "wav"
            return b"ok"

    vm = _DummyTts()
    vm.tts_adapter = FakeAdapter()

    out = vm.speak_to_bytes("## Hello **world**", format="wav")
    assert out == b"ok"
    assert captured["text"] == "Hello world"


def test_speak_to_file_sanitizes_markdown_by_default(tmp_path) -> None:
    captured: dict[str, str] = {}

    class FakeAdapter:
        def is_available(self) -> bool:
            return True

        def synthesize_to_file(self, text, output_path: str, *, format: str | None = None) -> str:
            captured["text"] = str(text)
            assert format is None
            return str(output_path)

    vm = _DummyTts()
    vm.tts_adapter = FakeAdapter()

    out_path = tmp_path / "out.wav"
    res = vm.speak_to_file("### Hello **world**", str(out_path))
    assert res == str(out_path)
    assert captured["text"] == "Hello world"
