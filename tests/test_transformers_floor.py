"""One Transformers floor check for the vendored Qwen3 models (ASR and TTS).

Torch-free: fake `torch` / `transformers` modules stand in, so these run on
every Python in CI, including 3.9 where Transformers 5 does not exist.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
from packaging.requirements import Requirement

from abstractvoice import transformers_floor as floor

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib


def _extra_floor(extra: str) -> str:
    pyproject = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())
    floors = [
        spec.version
        for line in pyproject["project"]["optional-dependencies"][extra]
        for req in [Requirement(line)]
        if req.name == "transformers" and req.marker is not None and not req.marker.evaluate({"python_version": "3.9"})
        for spec in req.specifier
        if spec.operator == ">="
    ]
    assert len(floors) == 1, floors
    return floors[0]


def test_floors_are_the_install_extras_floors():
    assert floor.QWEN3_ASR_MIN_TRANSFORMERS == _extra_floor("stt-hf")
    assert floor.QWEN3_TTS_MIN_TRANSFORMERS == _extra_floor("qwen3-tts")


def test_older_release_gets_a_plain_error():
    with pytest.raises(floor.UnsupportedTransformersVersion) as err:
        floor.require_transformers("5.9.0", model="Qwen3-TTS", extra="qwen3-tts", installed="5.8.1")
    message = str(err.value)
    assert message.startswith("Qwen3-TTS needs Transformers 5.9.0 or newer (Python 3.10+); transformers 5.8.1 is installed.")
    assert 'pip install -U "abstractvoice[qwen3-tts]"' in message


@pytest.mark.parametrize("installed", ["5.9.0", "5.9.0.dev0", "5.9.0rc1", "5.17.0", "6.0.0"])
def test_floor_and_newer_pass(installed):
    floor.require_transformers("5.9.0", model="Qwen3-TTS", extra="qwen3-tts", installed=installed)


def test_qwen3_tts_refuses_before_downloading_or_loading(monkeypatch):
    from abstractvoice.qwen3_tts.runtime import Qwen3TTSRuntime

    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    fake_tf = types.ModuleType("transformers")
    fake_tf.__version__ = "4.57.6"
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)

    runtime = Qwen3TTSRuntime(allow_downloads=False)
    monkeypatch.setattr(runtime, "snapshot_dir", lambda **_: pytest.fail("reached the snapshot / download step"))

    with pytest.raises(RuntimeError, match=r"^Qwen3-TTS needs Transformers 5\.9\.0 or newer"):
        runtime._ensure_loaded()
    assert not runtime.is_loaded
