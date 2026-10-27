"""`/save` keeps the LLM provider and the voice choices; `/load` restores them.

Before 0.19.0 a `.mem` file held the model only: loading it switched the model on whatever
provider was current, and the TTS engine, profile and cloned voice were lost.
"""

from __future__ import annotations

import json
from pathlib import Path


def _repl(provider_name: str = "lmstudio"):
    from abstractvoice.examples.cli_repl import VoiceREPL
    from abstractvoice.examples.llm_provider import resolve_provider

    class Profile:
        profile_id = "M3"

    class FakeVoiceManager:
        _tts_engine_name = "supertonic"
        _stt_engine_name = "faster_whisper"

        def __init__(self) -> None:
            self.speed = 1.0

        def get_language(self) -> str:
            return "fr"

        def get_active_profile(self, kind="tts"):
            return Profile()

        def get_speed(self) -> float:
            return self.speed

        def set_speed(self, value: float) -> None:
            self.speed = value

        def get_whisper(self) -> str:
            return "base"

        def set_whisper(self, value: str) -> None:
            pass

    repl = VoiceREPL.__new__(VoiceREPL)
    repl.voice_manager = FakeVoiceManager()
    repl.provider = resolve_provider(provider_name)
    repl.model = "qwen3.6-35b"
    repl.system_prompt = "be brief"
    repl.system_tokens = repl.user_tokens = repl.assistant_tokens = 0
    repl.temperature = 0.4
    repl.max_tokens = 4096
    repl.messages = [{"role": "system", "content": "be brief"}]
    repl.current_tts_voice = "voice-123"
    repl.debug_mode = False
    repl._initial_whisper_model = "base"
    return repl


def test_save_records_the_provider_and_the_voice(tmp_path: Path) -> None:
    repl = _repl("lmstudio")
    target = tmp_path / "chat"
    repl.do_save(str(target))
    data = json.loads((tmp_path / "chat.mem").read_text())
    assert data["header"]["provider"]["name"] == "lmstudio"
    assert data["header"]["provider"]["base_url"] == repl.provider.base_url
    assert data["voice"] == {
        "tts_engine": "supertonic",
        "language": "fr",
        "profile": "M3",
        "stt_engine": "faster_whisper",
        "cloned_voice": "voice-123",
    }


def test_load_restores_the_provider_then_the_model_and_reapplies_the_voice(tmp_path: Path, capsys) -> None:
    saved = _repl("lmstudio")
    saved.do_save(str(tmp_path / "chat"))

    repl = _repl("ollama")  # a session on another provider
    applied = []
    repl.do_tts_engine = lambda arg: applied.append(("engine", arg))
    repl.do_language = lambda arg: applied.append(("language", arg))
    repl.do_profile = lambda arg: applied.append(("profile", arg))
    repl.do_tts_voice = lambda arg: applied.append(("voice", arg))
    repl.do_stt_engine = lambda arg: applied.append(("stt", arg))
    repl._reset_and_recalculate_tokens = lambda: None
    repl.voice_manager._tts_engine_name = "piper"
    repl.voice_manager._stt_engine_name = "openai"
    repl.voice_manager.get_language = lambda: "en"
    repl.do_load(str(tmp_path / "chat"))

    assert repl.provider.name == "lmstudio" and repl.model == "qwen3.6-35b"
    assert applied == [
        ("engine", "supertonic"), ("language", "fr"), ("profile", "M3"), ("voice", "clone voice-123"), ("stt", "faster_whisper")
    ]
    out = capsys.readouterr().out
    assert "Provider changed from ollama" in out


def test_load_of_a_file_without_provider_keeps_the_current_one(tmp_path: Path) -> None:
    (tmp_path / "old.mem").write_text(json.dumps({"header": {"model": "m"}, "messages": [{"role": "user", "content": "hi"}]}))
    repl = _repl("ollama")
    repl._reset_and_recalculate_tokens = lambda: None
    repl.do_load(str(tmp_path / "old"))
    assert repl.provider.name == "ollama" and repl.model == "m"


def test_load_of_a_session_on_the_preset_voice_leaves_the_cloned_voice(tmp_path: Path) -> None:
    saved = _repl("lmstudio")
    saved.current_tts_voice = None
    saved.do_save(str(tmp_path / "preset"))
    repl = _repl("lmstudio")  # currently on a cloned voice
    applied = []
    repl.do_tts_engine = lambda arg: applied.append(("engine", arg))
    repl.do_language = lambda arg: applied.append(("language", arg))
    repl.do_profile = lambda arg: applied.append(("profile", arg))
    repl.do_tts_voice = lambda arg: applied.append(("voice", arg))
    repl.do_stt_engine = lambda arg: applied.append(("stt", arg))
    repl._reset_and_recalculate_tokens = lambda: None
    repl.do_load(str(tmp_path / "preset"))
    assert ("voice", "base") in applied
