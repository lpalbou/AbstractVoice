from __future__ import annotations


def test_interactive_auto_prefers_installed_supertonic(monkeypatch) -> None:
    import abstractvoice.examples.tts_defaults as defaults

    monkeypatch.setitem(defaults._INSTALL_CHECKS, "supertonic", lambda: True)
    monkeypatch.setitem(defaults._INSTALL_CHECKS, "piper", lambda: True)

    assert defaults.resolve_interactive_tts_engine("auto", language="en") == "supertonic"


def test_interactive_auto_falls_back_to_piper_without_supertonic(monkeypatch) -> None:
    import abstractvoice.examples.tts_defaults as defaults

    monkeypatch.setitem(defaults._INSTALL_CHECKS, "supertonic", lambda: False)
    monkeypatch.setitem(defaults._INSTALL_CHECKS, "piper", lambda: True)

    assert defaults.resolve_interactive_tts_engine("auto", language="en") == "piper"


def test_interactive_auto_falls_back_to_openai_without_local_runtime(monkeypatch) -> None:
    import abstractvoice.examples.tts_defaults as defaults

    monkeypatch.setitem(defaults._INSTALL_CHECKS, "supertonic", lambda: False)
    monkeypatch.setitem(defaults._INSTALL_CHECKS, "piper", lambda: False)

    assert defaults.resolve_interactive_tts_engine("auto", language="en") == "openai"


def test_interactive_explicit_openai_stays_remote() -> None:
    import abstractvoice.examples.tts_defaults as defaults

    assert defaults.resolve_interactive_tts_engine("openai", language="en") == "openai"


def test_interactive_explicit_supertonic_stays_supertonic() -> None:
    import abstractvoice.examples.tts_defaults as defaults

    assert defaults.resolve_interactive_tts_engine("supertonic", language="en") == "supertonic"
