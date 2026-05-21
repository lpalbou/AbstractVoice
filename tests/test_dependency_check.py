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


def test_dependency_checker_tracks_lightweight_base_and_local_profiles() -> None:
    checker = DependencyChecker(verbose=False)

    assert set(checker.CORE_DEPS) == {"numpy", "requests", "appdirs"}

    for package in (
        "piper-tts",
        "onnxruntime",
    ):
        assert package in checker.LOCAL_TTS_DEPS
        assert package not in checker.CORE_DEPS

    for package in (
        "huggingface_hub",
    ):
        assert package in checker.OPTIONAL_DEPS
        assert package not in checker.CORE_DEPS

    for package in (
        "faster-whisper",
        "soundfile",
    ):
        assert package in checker.LOCAL_STT_DEPS
        assert package not in checker.CORE_DEPS

    for package in (
        "sounddevice",
        "soundfile",
        "webrtcvad",
    ):
        assert package in checker.AUDIO_IO_DEPS
        assert package not in checker.CORE_DEPS

    assert checker.PYTORCH_COMPAT["torch"][1] is None
    assert checker.PYTORCH_COMPAT["torchvision"][1] is None


def test_base_dependencies_exclude_local_runtime_stacks() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    deps = pyproject["project"]["dependencies"]
    forbidden_prefixes = (
        "piper-tts",
        "huggingface_hub",
        "faster-whisper",
        "sounddevice",
        "soundfile",
        "webrtcvad",
        "torch",
        "torchaudio",
        "torchvision",
        "f5-tts",
        "omnivoice",
        "onnxruntime",
    )

    assert "numpy>=1.24.0" in deps
    assert "requests>=2.31.0" in deps
    assert "appdirs>=1.4.0" in deps
    for prefix in forbidden_prefixes:
        assert not _has_dep(deps, prefix)


def test_local_voice_extras_include_expected_runtime_stacks() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    extras = pyproject["project"]["optional-dependencies"]

    removed_aliases = {
        "voice",
        "voice-full",
        "local-tts",
        "local-stt",
        "core-stt",
        "audio-only",
        "legacy-stt",
        "all",
        "web-cloning",
        "web-audiodit",
        "web-omnivoice",
        "web-chroma",
        "web-full",
        "local",
    }
    assert removed_aliases.isdisjoint(extras)

    assert "piper-tts>=1.2.0" in extras["piper"]
    assert "onnxruntime>=1.19.0" in extras["supertonic"]

    assert "faster-whisper>=0.10.0" in extras["stt"]
    assert "soundfile>=0.12.1" in extras["stt"]

    for name in ("audio-io",):
        assert "sounddevice>=0.4.6" in extras[name]
        assert "webrtcvad>=2.0.10" in extras[name]
        assert "soundfile>=0.12.1" in extras[name]

    platform = extras["apple"]
    for dep in (
        "piper-tts>=1.2.0",
        "onnxruntime>=1.19.0",
        "faster-whisper>=0.10.0",
        "sounddevice>=0.4.6",
        "webrtcvad>=2.0.10",
        "soundfile>=0.12.1",
        "librosa>=0.10.0",
        "huggingface_hub>=0.20.0",
        "torch>=2.0.0",
        "safetensors>=0.4.0",
        "einops>=0.8.0",
        "sentencepiece>=0.1.99",
    ):
        assert dep in platform
    assert _has_marked_dep(platform, "f5-tts>=1.1.0", "python_version >= '3.10'")
    assert _has_marked_dep(platform, "omnivoice>=0.1.5", "python_version >= '3.10'")
    assert _has_marked_dep(platform, "aec-audio-processing>=1.0.1", "python_version >= '3.11'")
    assert extras["gpu"] == platform

    all_apple = extras["all-apple"]
    for dep in platform:
        assert dep in all_apple
    for dep in extras["web"]:
        assert dep in all_apple
    assert extras["all-gpu"] == all_apple


def test_remote_and_heavy_engine_extras_are_self_contained() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    extras = pyproject["project"]["optional-dependencies"]

    for name in ("openai", "openai-compatible", "remote"):
        assert extras[name] == []

    for name in (
        "cloning",
        "audiodit",
        "omnivoice",
        "chroma",
    ):
        assert _has_dep(extras[name], "soundfile>=0.12.1")
    assert _has_dep(extras["supertonic"], "onnxruntime>=1.19.0")


def test_python_support_contract_includes_39() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())

    assert pyproject["project"]["requires-python"] == ">=3.9"
    classifiers = pyproject["project"]["classifiers"]
    assert "Programming Language :: Python :: 3.9" in classifiers


def test_stt_extras_use_current_stack_and_remove_legacy_alias() -> None:
    pyproject = Path("pyproject.toml").read_text()

    stt_section = re.search(r"^stt = \[(.*?)^\]", pyproject, re.MULTILINE | re.DOTALL)
    legacy_stt_section = re.search(r"^legacy-stt = \[(.*?)^\]", pyproject, re.MULTILINE | re.DOTALL)
    core_stt_section = re.search(r"^core-stt = \[(.*?)^\]", pyproject, re.MULTILINE | re.DOTALL)

    assert stt_section is not None
    assert "faster-whisper>=0.10.0" in stt_section.group(1)
    assert "openai-whisper>=20230314" not in stt_section.group(1)
    assert core_stt_section is None
    assert legacy_stt_section is None


def test_web_extra_is_lightweight_and_composes_with_engine_extras() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    extras = pyproject["project"]["optional-dependencies"]

    assert "fastapi>=0.100.0" in extras["web"]
    assert "uvicorn>=0.23.0" in extras["web"]
    assert "python-multipart>=0.0.9" in extras["web"]

    assert not _has_dep(extras["web"], "omnivoice>=0.1.5")
    assert not _has_dep(extras["web"], "f5-tts>=1.1.0")
    assert not _has_dep(extras["web"], "torch")
    assert _has_dep(extras["omnivoice"], "omnivoice>=0.1.5")
    assert not any(dep.startswith("torchvision") for dep in extras["omnivoice"])


def test_python39_optional_engine_markers_are_resolver_safe() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    extras = pyproject["project"]["optional-dependencies"]

    assert "httpx>=0.23.0" in extras["test"]
    assert "soundfile>=0.12.1" in extras["test"]
    assert "webrtcvad>=2.0.10" in extras["test"]
    assert "transformers>=4.55.4,<5; python_version < '3.10'" in extras["audiodit"]
    assert "transformers>=5.4.0; python_version >= '3.10'" in extras["audiodit"]
    assert _has_marked_dep(extras["cloning"], "f5-tts>=1.1.0", "python_version >= '3.10'")
    assert _has_marked_dep(extras["apple"], "f5-tts>=1.1.0", "python_version >= '3.10'")
    assert _has_marked_dep(extras["apple"], "omnivoice>=0.1.5", "python_version >= '3.10'")
    assert all("python_version >= '3.10'" in dep for dep in extras["chroma"])
    assert all("python_version >= '3.10'" in dep for dep in extras["omnivoice"])
    assert "aec-audio-processing>=1.0.1; python_version >= '3.11'" in extras["aec"]
    assert "aec-audio-processing>=1.0.1; python_version >= '3.11'" in extras["apple"]
    assert "aec-audio-processing>=1.0.1; python_version >= '3.11'" in extras["gpu"]


def test_f5_runtime_guard_explains_python39(monkeypatch: pytest.MonkeyPatch) -> None:
    from abstractvoice.cloning.engine_f5 import F5TTSVoiceCloningEngine

    monkeypatch.setattr(sys, "version_info", (3, 9, 25))

    with pytest.raises(RuntimeError, match=r"OpenF5/F5-TTS cloning requires Python >=3\.10"):
        F5TTSVoiceCloningEngine()
