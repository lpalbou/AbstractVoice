from __future__ import annotations

import tomllib
from pathlib import Path

from abstractvoice.dependency_check import DependencyChecker


def test_dependency_checker_tracks_current_core_voice_stack() -> None:
    checker = DependencyChecker(verbose=False)

    for package in (
        "piper-tts",
        "huggingface_hub",
        "faster-whisper",
        "sounddevice",
        "soundfile",
        "webrtcvad",
    ):
        assert package in checker.CORE_DEPS

    assert checker.PYTORCH_COMPAT["torch"][1] is None
    assert checker.PYTORCH_COMPAT["torchvision"][1] is None


def test_stt_extras_use_current_stack_and_keep_legacy_alias() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    extras = pyproject["project"]["optional-dependencies"]

    assert "faster-whisper>=0.10.0" in extras["stt"]
    assert "openai-whisper>=20230314" not in extras["stt"]
    assert "faster-whisper>=0.10.0" in extras["core-stt"]
    assert "openai-whisper>=20230314" in extras["legacy-stt"]
