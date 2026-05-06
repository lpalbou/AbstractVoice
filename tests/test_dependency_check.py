from __future__ import annotations

from pathlib import Path
import re
import sys

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

from abstractvoice.dependency_check import DependencyChecker


def _has_dep(deps: list[str], prefix: str) -> bool:
    return any(dep.startswith(prefix) for dep in deps)


def _has_marked_dep(deps: list[str], prefix: str, marker: str) -> bool:
    return any(dep.startswith(prefix) and marker in dep for dep in deps)


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


def test_python_support_contract_includes_39() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())

    assert pyproject["project"]["requires-python"] == ">=3.9"
    classifiers = pyproject["project"]["classifiers"]
    assert "Programming Language :: Python :: 3.9" in classifiers


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

    assert not _has_dep(extras["web"], "omnivoice>=0.1.2")
    assert _has_dep(extras["omnivoice"], "omnivoice>=0.1.2")
    assert _has_dep(extras["web-omnivoice"], "omnivoice>=0.1.2")
    assert _has_dep(extras["web-full"], "omnivoice>=0.1.2")
    assert _has_marked_dep(extras["web-cloning"], "f5-tts>=1.1.0", "python_version >= '3.10'")
    assert not any(dep.startswith("torchvision") for dep in extras["omnivoice"])
    assert not any(dep.startswith("torchvision") for dep in extras["web-omnivoice"])


def test_python39_optional_engine_markers_are_resolver_safe() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    extras = pyproject["project"]["optional-dependencies"]

    assert "httpx>=0.23.0" in extras["test"]
    assert "transformers>=4.55.4,<5; python_version < '3.10'" in extras["audiodit"]
    assert "transformers>=5.3.0; python_version >= '3.10'" in extras["audiodit"]
    assert _has_marked_dep(extras["cloning"], "f5-tts>=1.1.0", "python_version >= '3.10'")
    assert _has_marked_dep(extras["web-cloning"], "f5-tts>=1.1.0", "python_version >= '3.10'")
    assert _has_marked_dep(extras["web-full"], "f5-tts>=1.1.0", "python_version >= '3.10'")
    assert _has_marked_dep(extras["all"], "f5-tts>=1.1.0", "python_version >= '3.10'")
    assert all("python_version >= '3.10'" in dep for dep in extras["chroma"])
    assert all("python_version >= '3.10'" in dep for dep in extras["omnivoice"])
    assert "aec-audio-processing>=1.0.1; python_version >= '3.11'" in extras["aec"]


def test_f5_runtime_guard_explains_python39(monkeypatch: pytest.MonkeyPatch) -> None:
    from abstractvoice.cloning.engine_f5 import F5TTSVoiceCloningEngine

    monkeypatch.setattr(sys, "version_info", (3, 9, 25))

    with pytest.raises(RuntimeError, match=r"OpenF5/F5-TTS cloning requires Python >=3\.10"):
        F5TTSVoiceCloningEngine()
