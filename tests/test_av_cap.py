"""PyAV stays below 19 wherever faster-whisper is installed (framework rehearsal 0.6.3).

PyAV 19.0.0 (2026-09-29) removed the ``metadata_errors`` option of ``av.open``; faster-whisper
1.2.1 passes it when it decodes a file, so every file transcription failed with
``open() got an unexpected keyword argument 'metadata_errors'``. faster-whisper itself only asks
for ``av>=11``, so each extra that installs faster-whisper must carry the cap: the installer's
``abstractvoice[supertonic,stt]`` gateway voice stack reaches av through ``stt`` alone.
"""

from __future__ import annotations

import wave
from pathlib import Path

import pytest
from packaging.requirements import Requirement

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib


def _extras() -> dict:
    pyproject = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())
    return pyproject["project"]["optional-dependencies"]


def _faster_whisper_extras() -> list:
    return sorted(
        name
        for name, lines in _extras().items()
        if any(Requirement(line).name == "faster-whisper" for line in lines)
    )


def test_the_installer_voice_extra_installs_faster_whisper():
    # The check below must not pass vacuously: `stt` is the gateway installer's voice extra.
    assert {"stt", "apple", "gpu", "all-apple", "all-gpu"} <= set(_faster_whisper_extras())


@pytest.mark.parametrize("python_version", ["3.10", "3.11", "3.12", "3.13", "3.14"])
@pytest.mark.parametrize("extra", _faster_whisper_extras())
def test_every_faster_whisper_extra_refuses_pyav_19(extra, python_version):
    env = {"python_version": python_version, "python_full_version": f"{python_version}.0"}
    av_specs = [
        req.specifier
        for line in _extras()[extra]
        for req in [Requirement(line)]
        if req.name == "av" and (req.marker is None or req.marker.evaluate(env))
    ]
    assert av_specs, f"[{extra}] installs faster-whisper but does not constrain av on Python {python_version}"
    for spec in av_specs:
        assert not spec.contains("19.0.0"), f"[{extra}] allows av 19.0.0 ({spec})"
    assert any(spec.contains("18.1.0") for spec in av_specs), f"[{extra}] refuses av 18.1.0"


def test_faster_whisper_decodes_a_file_with_the_installed_pyav(tmp_path):
    av = pytest.importorskip("av")
    audio = pytest.importorskip("faster_whisper.audio")
    np = pytest.importorskip("numpy")

    path = tmp_path / "tone.wav"
    samples = (np.sin(np.linspace(0, 2 * np.pi * 440, 16000)) * 12000).astype("<i2")
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(16000)
        out.writeframes(samples.tobytes())

    decoded = audio.decode_audio(str(path), sampling_rate=16000)
    assert abs(len(decoded) - 16000) < 400, (av.__version__, len(decoded))
