from __future__ import annotations

from pathlib import Path
import re

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

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


def test_web_engine_extras_are_explicit_install_bundles() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    extras = pyproject["project"]["optional-dependencies"]

    for name in ("web", "web-cloning", "web-audiodit", "web-omnivoice", "web-chroma", "web-full"):
        assert name in extras
        assert "fastapi>=0.100.0" in extras[name]
        assert "uvicorn>=0.23.0" in extras[name]
        assert "python-multipart>=0.0.9" in extras[name]

    assert "omnivoice>=0.1.2" not in extras["web"]
    assert "omnivoice>=0.1.2" in extras["omnivoice"]
    assert "omnivoice>=0.1.2" in extras["web-omnivoice"]
    assert "omnivoice>=0.1.2" in extras["web-full"]
    assert "f5-tts>=1.1.0" in extras["web-cloning"]
    assert not any(dep.startswith("torchvision") for dep in extras["omnivoice"])
    assert not any(dep.startswith("torchvision") for dep in extras["web-omnivoice"])
