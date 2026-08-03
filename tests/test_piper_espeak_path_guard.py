"""espeak-ng's fixed path buffer must never terminate the host process.

espeak-ng (as vendored by piper's wheels, <= 1.52) keeps its data path in
``char path_home[160]`` on POSIX. A bundled data dir at or over that limit is
silently rejected; espeak falls back to the path compiled in on piper's build
machine and, failing to find it, prints an error and exits the interpreter.
Measured boundary: 159 chars works, 160 kills the process.

These tests exercise the guard without piper installed and without espeak ever
running: the guard's whole point is to decide things *before* the C library is
involved.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from abstractvoice.adapters.tts_piper import (
    PiperTTSAdapter,
    _ESPEAK_PATH_HOME_LIMIT,
    _espeak_data_dir_override,
)


@pytest.fixture()
def deep_data_dir(tmp_path) -> Path:
    """A real directory whose absolute path is over the espeak limit."""
    pad = "d" * max(1, _ESPEAK_PATH_HOME_LIMIT - len(str(tmp_path)) - 2)
    deep = tmp_path / pad / "espeak-ng-data"
    deep.mkdir(parents=True)
    (deep / "phontab").write_bytes(b"x")
    assert len(str(deep)) >= _ESPEAK_PATH_HOME_LIMIT
    return deep


def test_a_fitting_path_is_left_alone(tmp_path):
    data = tmp_path / "espeak-ng-data"
    data.mkdir()

    assert _espeak_data_dir_override(data) is None


def test_an_over_limit_path_is_aliased_through_a_short_symlink(deep_data_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "h"))

    alias = _espeak_data_dir_override(deep_data_dir)

    assert alias is not None
    assert len(str(alias)) < _ESPEAK_PATH_HOME_LIMIT
    assert alias.is_symlink()
    assert alias.resolve() == deep_data_dir.resolve()
    assert (alias / "phontab").read_bytes() == b"x"  # espeak reads through it


def test_the_alias_is_stable_and_a_stale_one_is_repointed(deep_data_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "h"))

    first = _espeak_data_dir_override(deep_data_dir)
    second = _espeak_data_dir_override(deep_data_dir)
    assert first == second, "one install, one alias"

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    first.unlink()
    first.symlink_to(elsewhere, target_is_directory=True)

    repointed = _espeak_data_dir_override(deep_data_dir)
    assert repointed.resolve() == deep_data_dir.resolve()


def test_when_no_short_alias_is_possible_the_guard_raises_instead_of_dying(deep_data_dir, tmp_path, monkeypatch):
    import tempfile

    deep_home = tmp_path / ("h" * max(1, _ESPEAK_PATH_HOME_LIMIT - len(str(tmp_path)) - 2))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: deep_home))
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(deep_home))

    with pytest.raises(RuntimeError) as err:
        _espeak_data_dir_override(deep_data_dir)

    message = str(err.value)
    assert "espeak-ng" in message
    assert str(_ESPEAK_PATH_HOME_LIMIT) in message
    assert "shorter path" in message


def _adapter_with_fake_piper(monkeypatch, tmp_path, data_dir: Path, *, load_accepts_alias: bool):
    """A PiperTTSAdapter wired to a fake piper install at `data_dir`."""
    fake_phonemize = types.ModuleType("piper.phonemize_espeak")
    fake_phonemize.ESPEAK_DATA_DIR = data_dir
    fake_piper = types.ModuleType("piper")
    fake_piper.phonemize_espeak = fake_phonemize
    monkeypatch.setitem(sys.modules, "piper", fake_piper)
    monkeypatch.setitem(sys.modules, "piper.phonemize_espeak", fake_phonemize)

    class _FakeVoice:
        if load_accepts_alias:
            @staticmethod
            def load(model_path, config_path, use_cuda=False, espeak_data_dir=None):
                return object()
        else:
            @staticmethod
            def load(model_path, config_path, use_cuda=False):
                return object()

    adapter = PiperTTSAdapter(language="en", model_dir=str(tmp_path / "models"), auto_load=False)
    adapter._PiperVoice = _FakeVoice
    adapter._piper_available = True
    return adapter


def test_load_kwargs_are_empty_for_a_normal_install(monkeypatch, tmp_path):
    data = tmp_path / "espeak-ng-data"
    data.mkdir()
    adapter = _adapter_with_fake_piper(monkeypatch, tmp_path, data, load_accepts_alias=True)

    assert adapter._espeak_load_kwargs() == {}


def test_load_kwargs_carry_the_alias_for_a_deep_install(monkeypatch, tmp_path, deep_data_dir):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "h"))
    adapter = _adapter_with_fake_piper(monkeypatch, tmp_path, deep_data_dir, load_accepts_alias=True)

    kwargs = adapter._espeak_load_kwargs()

    assert set(kwargs) == {"espeak_data_dir"}
    assert Path(kwargs["espeak_data_dir"]).resolve() == deep_data_dir.resolve()
    assert len(kwargs["espeak_data_dir"]) < _ESPEAK_PATH_HOME_LIMIT


def test_a_piper_that_cannot_take_an_alias_gets_a_clear_error_not_a_dead_process(
    monkeypatch, tmp_path, deep_data_dir
):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "h"))
    adapter = _adapter_with_fake_piper(monkeypatch, tmp_path, deep_data_dir, load_accepts_alias=False)

    with pytest.raises(RuntimeError) as err:
        adapter._espeak_load_kwargs()

    assert "espeak_data_dir" in str(err.value)
