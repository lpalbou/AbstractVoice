from __future__ import annotations

from pathlib import Path
import re

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
    pyproject = Path("pyproject.toml").read_text()

    stt_section = re.search(r"^stt = \[(.*?)^\]", pyproject, re.MULTILINE | re.DOTALL)
    core_stt_section = re.search(r"^core-stt = \[(.*?)^\]", pyproject, re.MULTILINE | re.DOTALL)
    legacy_stt_section = re.search(r"^legacy-stt = \[(.*?)^\]", pyproject, re.MULTILINE | re.DOTALL)

    assert stt_section is not None
    assert core_stt_section is not None
    assert legacy_stt_section is not None
    assert "faster-whisper>=0.10.0" in stt_section.group(1)
    assert "openai-whisper>=20230314" not in stt_section.group(1)
    assert "faster-whisper>=0.10.0" in core_stt_section.group(1)
    assert "openai-whisper>=20230314" in legacy_stt_section.group(1)
