from __future__ import annotations


def test_one_shot_tts_uses_explicit_provider_model_and_voice_profile(tmp_path, capsys) -> None:
    from abstractvoice.examples import voice_cli

    output = tmp_path / "hello.wav"
    args = voice_cli.parse_args(
        [
            "--provider",
            "openai",
            "--model",
            "tts-1",
            "--voice",
            "nova",
            "--prompt",
            "Hello.",
            "--output",
            str(output),
        ]
    )
    created: dict = {}
    calls: dict = {}

    class FakeVoiceManager:
        def __init__(self, **kwargs) -> None:
            created.update(kwargs)

        def set_profile(self, profile_id: str, *, kind: str = "tts") -> bool:
            calls["profile"] = {"profile_id": profile_id, "kind": kind}
            return True

        def speak_to_file(self, text: str, output_path: str, format=None, voice=None) -> str:
            calls["speak"] = {
                "text": text,
                "output_path": output_path,
                "format": format,
                "voice": voice,
            }
            return output_path

        def cleanup(self) -> None:
            calls["cleanup"] = True

    result = voice_cli._run_one_shot_tts(args, voice_manager_factory=FakeVoiceManager)

    assert result == str(output)
    assert created["tts_engine"] == "openai"
    assert created["tts_model"] == "tts-1"
    assert calls["profile"] == {"profile_id": "nova", "kind": "tts"}
    assert calls["speak"] == {
        "text": "Hello.",
        "output_path": str(output),
        "format": None,
        "voice": None,
    }
    assert calls["cleanup"] is True
    assert f"Wrote {output}" in capsys.readouterr().out


def test_one_shot_tts_defaults_do_not_leak_repl_llm_provider_or_model(tmp_path) -> None:
    from abstractvoice.examples import voice_cli

    output = tmp_path / "default.wav"
    args = voice_cli.parse_args(["--prompt", "Hello.", "--output", str(output)])
    created: dict = {}

    class FakeVoiceManager:
        def __init__(self, **kwargs) -> None:
            created.update(kwargs)

        def speak_to_file(self, text: str, output_path: str, format=None, voice=None) -> str:
            return output_path

        def cleanup(self) -> None:
            pass

    voice_cli._run_one_shot_tts(args, voice_manager_factory=FakeVoiceManager)

    assert args.provider_explicit is False
    assert args.model_explicit is False
    assert created["tts_engine"] == "auto"
    assert created["tts_model"] is None


def test_one_shot_tts_falls_back_to_cloned_voice_when_profile_not_found(tmp_path) -> None:
    from abstractvoice.examples import voice_cli

    output = tmp_path / "clone.wav"
    args = voice_cli.parse_args(
        [
            "--tts-engine",
            "piper",
            "--voice",
            "clone_alice",
            "--prompt",
            "Hello.",
            "--output",
            str(output),
        ]
    )
    calls: dict = {}

    class FakeVoiceManager:
        def __init__(self, **kwargs) -> None:
            calls["created"] = kwargs

        def set_profile(self, profile_id: str, *, kind: str = "tts") -> bool:
            calls["profile"] = {"profile_id": profile_id, "kind": kind}
            return False

        def list_cloned_voices(self):
            return [{"voice_id": "clone-id-123", "name": "clone_alice"}]

        def speak_to_file(self, text: str, output_path: str, format=None, voice=None) -> str:
            calls["speak"] = {"text": text, "output_path": output_path, "voice": voice}
            return output_path

        def cleanup(self) -> None:
            pass

    voice_cli._run_one_shot_tts(args, voice_manager_factory=FakeVoiceManager)

    assert calls["created"]["tts_engine"] == "piper"
    assert calls["profile"] == {"profile_id": "clone_alice", "kind": "tts"}
    assert calls["speak"] == {
        "text": "Hello.",
        "output_path": str(output),
        "voice": "clone-id-123",
    }


def test_one_shot_tts_provider_url_maps_to_openai_compatible_base_url(tmp_path) -> None:
    from abstractvoice.examples import voice_cli

    output = tmp_path / "remote.wav"
    args = voice_cli.parse_args(
        [
            "--provider",
            "http://remote.test/v1",
            "--model",
            "remote-tts",
            "--prompt",
            "Hello.",
            "--output",
            str(output),
        ]
    )
    created: dict = {}

    class FakeVoiceManager:
        def __init__(self, **kwargs) -> None:
            created.update(kwargs)

        def speak_to_file(self, text: str, output_path: str, format=None, voice=None) -> str:
            return output_path

        def cleanup(self) -> None:
            pass

    voice_cli._run_one_shot_tts(args, voice_manager_factory=FakeVoiceManager)

    assert created["tts_engine"] == "openai-compatible"
    assert created["tts_model"] == "remote-tts"
    assert created["remote_base_url"] == "http://remote.test/v1"
