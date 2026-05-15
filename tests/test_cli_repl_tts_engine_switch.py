from __future__ import annotations

import pytest


def test_repl_speak_unavailable_supertonic_uses_supertonic_message() -> None:
    from abstractvoice.examples.cli_repl import VoiceREPL

    class FakeAdapter:
        engine_id = "supertonic"

        def is_available(self) -> bool:
            return False

        def get_unavailable_reason(self) -> str:
            return (
                "Supertonic 3 artifacts are not available locally.\n"
                "Run: python -m abstractvoice download --supertonic"
            )

    class FakeVoiceManager:
        tts_adapter = FakeAdapter()

    repl = VoiceREPL.__new__(VoiceREPL)
    repl.voice_manager = FakeVoiceManager()
    repl.current_tts_voice = None
    repl.current_language = "en"
    repl._debug_save_wav = False

    with pytest.raises(RuntimeError) as exc:
        repl._speak_with_spinner_until_audio_starts("hello")

    msg = str(exc.value)
    assert "Supertonic 3 artifacts" in msg
    assert "download --supertonic" in msg
    assert "Piper voice model" not in msg


def test_repl_tts_engine_switch_resets_clone_and_reports_default_profile(capsys) -> None:
    from abstractvoice.examples.cli_repl import VoiceREPL

    class Profile:
        profile_id = "M1"
        label = "Supertonic M1"

    class FakeVoiceManager:
        def __init__(self) -> None:
            self.calls = []

        def set_tts_engine(self, engine: str, *, tts_model=None) -> str:
            self.calls.append({"engine": engine, "tts_model": tts_model})
            return engine

        def get_active_profile(self, kind="tts"):
            return Profile()

    vm = FakeVoiceManager()
    repl = VoiceREPL.__new__(VoiceREPL)
    repl.voice_manager = vm
    repl.current_language = "en"
    repl.current_tts_voice = "clone_a"
    repl._initial_tts_model = None
    repl._initial_stt_model = None
    repl._initial_stt_engine = "openai"
    repl.debug_mode = False
    repl.cloning_engine = "f5_tts"
    repl.remote_base_url = None
    repl.remote_api_key = None
    repl.remote_timeout_s = None

    repl.do_tts_engine("supertonic")

    assert vm.calls == [{"engine": "supertonic", "tts_model": None}]
    assert repl.current_tts_voice is None
    assert repl._initial_tts_engine == "supertonic"
    out = capsys.readouterr().out
    assert "TTS engine set to: supertonic" in out
    assert "profile: M1" in out


def test_repl_tts_engine_rejects_local_policy_alias(capsys) -> None:
    from abstractvoice.examples.cli_repl import VoiceREPL

    class FakeVoiceManager:
        def __init__(self) -> None:
            self.calls = []

        def set_tts_engine(self, engine: str, *, tts_model=None) -> str:
            self.calls.append({"engine": engine, "tts_model": tts_model})
            return engine

    vm = FakeVoiceManager()
    repl = VoiceREPL.__new__(VoiceREPL)
    repl.voice_manager = vm
    repl.current_language = "en"
    repl.current_tts_voice = None
    repl._initial_tts_model = None
    repl._initial_stt_model = None
    repl._initial_stt_engine = "openai"
    repl.debug_mode = False
    repl.cloning_engine = "f5_tts"
    repl.remote_base_url = None
    repl.remote_api_key = None
    repl.remote_timeout_s = None

    repl.do_tts_engine("local")

    assert vm.calls == []
    out = capsys.readouterr().out
    assert "Usage: /tts_engine auto|supertonic|piper|openai|openai-compatible|audiodit|omnivoice" in out
    assert "local" not in out
