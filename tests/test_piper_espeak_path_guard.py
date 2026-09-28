"""espeak-ng's fixed path buffer must never terminate the host process.

espeak-ng (as vendored by piper's wheels, <= 1.52) keeps its data path in
``char path_home[160]`` on POSIX. A bundled data dir at or over that limit is
silently rejected; espeak falls back to the path compiled in on piper's build
machine and, failing to find it, prints an error and exits the interpreter.
Measured boundary: 159 chars works, 160 kills the process.

These tests exercise the guard without piper installed and without espeak ever
running: the guard's whole point is to decide things *before* the C library is
involved. They pass an explicit limit measured from `tmp_path`, so they do not
depend on how long TMPDIR is (a deep TMPDIR puts every tmp path over 160).
"""

from __future__ import annotations

import sys
import types
from functools import partial
from pathlib import Path

import pytest

import abstractvoice.adapters.tts_piper as tts_piper
from abstractvoice.adapters.tts_piper import PiperTTSAdapter, _espeak_data_dir_override

# An alias under `tmp_path / "h"` (h/.cache/abstractvoice/espeak-<10 hex>) adds 40
# characters; the limit leaves room for it and for `tmp_path/espeak-ng-data`.
_ALIAS_ROOM = 60


@pytest.fixture()
def limit(tmp_path) -> int:
    """The espeak path limit these tests apply, measured from `tmp_path`."""
    return len(str(tmp_path)) + _ALIAS_ROOM


@pytest.fixture()
def deep_data_dir(tmp_path, limit) -> Path:
    """A real directory whose absolute path is over the limit."""
    deep = tmp_path / ("d" * _ALIAS_ROOM) / "espeak-ng-data"
    deep.mkdir(parents=True)
    (deep / "phontab").write_bytes(b"x")
    assert len(str(deep)) >= limit
    return deep


def test_a_fitting_path_is_left_alone(tmp_path, limit):
    data = tmp_path / "espeak-ng-data"
    data.mkdir()

    assert _espeak_data_dir_override(data, limit=limit) is None
    # The boundary itself: one character less than the path is already over it.
    assert _espeak_data_dir_override(data, limit=len(str(data)) + 1) is None


def test_an_over_limit_path_is_aliased_through_a_short_symlink(deep_data_dir, tmp_path, limit, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "h"))

    alias = _espeak_data_dir_override(deep_data_dir, limit=limit)

    assert alias is not None
    assert len(str(alias)) < limit
    assert alias.is_symlink()
    assert alias.resolve() == deep_data_dir.resolve()
    assert (alias / "phontab").read_bytes() == b"x"  # espeak reads through it


def test_the_alias_is_stable_and_a_stale_one_is_repointed(deep_data_dir, tmp_path, limit, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "h"))

    first = _espeak_data_dir_override(deep_data_dir, limit=limit)
    second = _espeak_data_dir_override(deep_data_dir, limit=limit)
    assert first == second, "one install, one alias"

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    first.unlink()
    first.symlink_to(elsewhere, target_is_directory=True)

    repointed = _espeak_data_dir_override(deep_data_dir, limit=limit)
    assert repointed.resolve() == deep_data_dir.resolve()


def test_when_no_short_alias_is_possible_the_guard_raises_instead_of_dying(deep_data_dir, tmp_path, limit, monkeypatch):
    import tempfile

    deep_home = tmp_path / ("h" * _ALIAS_ROOM)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: deep_home))
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(deep_home))

    with pytest.raises(RuntimeError) as err:
        _espeak_data_dir_override(deep_data_dir, limit=limit)

    message = str(err.value)
    assert "espeak-ng" in message
    assert f"limit {limit}" in message
    assert "shorter path" in message


def _adapter_with_fake_piper(monkeypatch, tmp_path, data_dir: Path, *, load_accepts_alias: bool, limit: int):
    """A PiperTTSAdapter wired to a fake piper install at `data_dir`, guarded at `limit`."""
    monkeypatch.setattr(tts_piper, "_espeak_data_dir_override", partial(_espeak_data_dir_override, limit=limit))
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
    adapter = _adapter_with_fake_piper(
        monkeypatch, tmp_path, data, load_accepts_alias=True, limit=len(str(data)) + 1
    )

    assert adapter._espeak_load_kwargs() == {}


def test_load_kwargs_carry_the_alias_for_a_deep_install(monkeypatch, tmp_path, deep_data_dir, limit):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "h"))
    adapter = _adapter_with_fake_piper(monkeypatch, tmp_path, deep_data_dir, load_accepts_alias=True, limit=limit)

    kwargs = adapter._espeak_load_kwargs()

    assert set(kwargs) == {"espeak_data_dir"}
    assert Path(kwargs["espeak_data_dir"]).resolve() == deep_data_dir.resolve()
    assert len(kwargs["espeak_data_dir"]) < limit


def test_a_piper_that_cannot_take_an_alias_gets_a_clear_error_not_a_dead_process(
    monkeypatch, tmp_path, deep_data_dir, limit
):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "h"))
    adapter = _adapter_with_fake_piper(monkeypatch, tmp_path, deep_data_dir, load_accepts_alias=False, limit=limit)

    with pytest.raises(RuntimeError) as err:
        adapter._espeak_load_kwargs()

    assert "PiperVoice.load has no espeak_data_dir parameter" in str(err.value)
