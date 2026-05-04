from __future__ import annotations

from abstractvoice.vm import tts_mixin as tts_mixin_module
from abstractvoice.vm.tts_mixin import TtsMixin


class _FakeRecognizer:
    def __init__(self) -> None:
        self.stopped = False

    def stop(self) -> bool:
        self.stopped = True
        return True


class _FakeAdapter:
    def __init__(self, *, engine_id: str = "piper") -> None:
        self.engine_id = engine_id
        self.language = None

    def set_language(self, language: str) -> bool:
        self.language = language
        return True


class _DummyVoiceManager(TtsMixin):
    LANGUAGES = {
        "en": {"name": "English"},
        "fr": {"name": "French"},
    }

    def __init__(
        self,
        *,
        language: str = "en",
        adapter: _FakeAdapter | None = None,
        recognizer: _FakeRecognizer | None = None,
        tts_engine_preference: str = "piper",
        debug_mode: bool = False,
    ) -> None:
        self.allow_downloads = False
        self.debug_mode = debug_mode
        self.language = language
        self.speed = 1.0
        self.tts_adapter = adapter
        self.tts_engine = object()
        self.voice_recognizer = recognizer
        self._tts_engine_name = None
        self._tts_engine_preference = tts_engine_preference

    def stop_speaking(self) -> bool:
        return True

    def _wire_tts_callbacks(self) -> None:
        return None


def test_set_language_discards_stopped_recognizer_so_listen_rebuilds_with_new_language() -> None:
    recognizer = _FakeRecognizer()
    adapter = _FakeAdapter(engine_id="piper")
    vm = _DummyVoiceManager(adapter=adapter, recognizer=recognizer)

    assert vm.set_language("fr") is True

    assert recognizer.stopped is True
    assert vm.voice_recognizer is None
    assert adapter.language == "fr"
    assert vm.language == "fr"


def test_set_language_same_non_catalog_language_debug_message_does_not_crash(capsys) -> None:
    vm = _DummyVoiceManager(
        language="eo",
        adapter=_FakeAdapter(engine_id="omnivoice"),
        tts_engine_preference="omnivoice",
        debug_mode=True,
    )

    assert vm.set_language("eo") is True
    assert "Already using eo voice" in capsys.readouterr().out


def test_set_language_allows_non_catalog_language_for_explicit_non_piper_engine(monkeypatch) -> None:
    adapter = _FakeAdapter(engine_id="omnivoice")
    vm = _DummyVoiceManager(adapter=None, tts_engine_preference="omnivoice")

    def fake_create_tts_adapter(**kwargs):
        assert kwargs["engine"] == "omnivoice"
        assert kwargs["language"] == "eo"
        return adapter, "omnivoice"

    monkeypatch.setattr(tts_mixin_module, "create_tts_adapter", fake_create_tts_adapter)

    assert vm.set_language("eo") is True
    assert adapter.language == "eo"
    assert vm.language == "eo"
