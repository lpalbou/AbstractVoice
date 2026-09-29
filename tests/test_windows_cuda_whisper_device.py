"""faster-whisper on Windows: CUDA only when CUDA 12 cuBLAS really loads (framework backlog 0988).

The platform is simulated (`sys.platform = "win32"`, a fake `os.add_dll_directory`, a fake
`nvidia` namespace package on disk, a fake `ctranslate2` module and a fake `ctypes.WinDLL`), so
these run on any OS.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture()
def wc(monkeypatch: pytest.MonkeyPatch):
    from abstractvoice.compute import windows_cuda

    monkeypatch.setattr(windows_cuda, "_prepared", None)
    monkeypatch.setattr(windows_cuda, "_cublas12", None)
    monkeypatch.setattr(windows_cuda, "_handles", [])
    monkeypatch.delenv("ABSTRACTVOICE_WHISPER_DEVICE", raising=False)
    return windows_cuda


def _fake_nvidia(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, libs=("cublas", "cuda_runtime")) -> list[Path]:
    root = tmp_path / "site-packages" / "nvidia"
    bins = []
    for name in libs:
        folder = root / name / "bin"
        folder.mkdir(parents=True)
        bins.append(folder)
    spec = importlib.util.spec_from_loader("nvidia", loader=None, is_package=True)
    spec.submodule_search_locations = [str(root)]
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name, *a, **k: spec if name == "nvidia" else real_find_spec(name, *a, **k)
    )
    return bins


def _fake_ctranslate2(monkeypatch: pytest.MonkeyPatch, cuda_devices: int) -> None:
    module = types.ModuleType("ctranslate2")
    module.get_cuda_device_count = lambda: cuda_devices  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ctranslate2", module)


def _simulate_windows(monkeypatch: pytest.MonkeyPatch, loadable: set[str]) -> list[str]:
    added: list[str] = []
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(os, "add_dll_directory", lambda p: added.append(p) or object(), raising=False)
    import ctypes

    def fake_windll(name, winmode=None):
        if name not in loadable:
            raise OSError(f"[WinError 126] {name} not found")
        return object()

    monkeypatch.setattr(ctypes, "WinDLL", fake_windll, raising=False)
    monkeypatch.setenv("PATH", "C:\\Windows\\system32")
    return added


def test_windows_cuda_with_cublas12_picks_cuda_and_puts_the_wheel_bins_on_the_dll_search(
    wc, tmp_path, monkeypatch
) -> None:
    from abstractvoice.compute.device import best_faster_whisper_device

    bins = _fake_nvidia(tmp_path, monkeypatch)
    _fake_ctranslate2(monkeypatch, cuda_devices=1)
    added = _simulate_windows(monkeypatch, loadable={"cublas64_12.dll"})

    assert best_faster_whisper_device() == "cuda"
    assert added == [str(b) for b in bins]
    path = os.environ["PATH"].split(os.pathsep)
    assert path[: len(bins)] == [str(b) for b in reversed(bins)]


def test_windows_cuda_without_cublas12_falls_back_to_cpu(wc, tmp_path, monkeypatch, caplog) -> None:
    from abstractvoice.compute.device import best_faster_whisper_device

    _fake_ctranslate2(monkeypatch, cuda_devices=1)
    _simulate_windows(monkeypatch, loadable=set())

    with caplog.at_level("WARNING"):
        assert best_faster_whisper_device() == "cpu"
    assert "cublas64_12.dll" in caplog.text


def test_windows_without_cuda_stays_on_cpu_without_probing_cublas(wc, monkeypatch) -> None:
    from abstractvoice.compute.device import best_faster_whisper_device

    _fake_ctranslate2(monkeypatch, cuda_devices=0)
    _simulate_windows(monkeypatch, loadable={"cublas64_12.dll"})

    assert best_faster_whisper_device() == "cpu"
    assert wc._cublas12 is None


def test_forced_device_is_honoured_on_windows(wc, monkeypatch) -> None:
    from abstractvoice.compute.device import best_faster_whisper_device

    _simulate_windows(monkeypatch, loadable=set())
    monkeypatch.setenv("ABSTRACTVOICE_WHISPER_DEVICE", "cuda:0")
    assert best_faster_whisper_device() == "cuda"


def test_off_windows_nothing_is_added_and_cublas_is_not_probed(wc, tmp_path, monkeypatch) -> None:
    from abstractvoice.compute.device import best_faster_whisper_device

    _fake_nvidia(tmp_path, monkeypatch)
    _fake_ctranslate2(monkeypatch, cuda_devices=1)
    monkeypatch.setattr(sys, "platform", "linux")
    before = os.environ.get("PATH")

    assert best_faster_whisper_device() == "cuda"
    assert wc.prepare_windows_cuda_dlls() == []
    assert os.environ.get("PATH") == before
    assert wc._cublas12 is None


def test_windows_without_nvidia_wheels_adds_nothing(wc, monkeypatch) -> None:
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name, *a, **k: None if name == "nvidia" else real_find_spec(name, *a, **k)
    )
    added = _simulate_windows(monkeypatch, loadable=set())
    assert wc.prepare_windows_cuda_dlls() == []
    assert added == []
