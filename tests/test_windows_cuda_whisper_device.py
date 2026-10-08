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
    monkeypatch.setattr(windows_cuda, "_cudnn9", None)
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


CUDA_TYPES = {"float32", "int8", "int8_float32", "int8_float16", "float16", "bfloat16", "int8_bfloat16"}


def _fake_ctranslate2(monkeypatch: pytest.MonkeyPatch, cuda_devices: int, cuda_types=CUDA_TYPES) -> None:
    module = types.ModuleType("ctranslate2")
    module.get_cuda_device_count = lambda: cuda_devices  # type: ignore[attr-defined]

    def supported(device, index=0):
        if device == "cpu":
            return {"int8", "int8_float32", "float32"}
        if device == "cuda" and cuda_devices > 0:
            return set(cuda_types)
        raise ValueError("This CTranslate2 package was not compiled with CUDA support")

    module.get_supported_compute_types = supported  # type: ignore[attr-defined]
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
    added = _simulate_windows(monkeypatch, loadable={"cublas64_12.dll", "cudnn_ops64_9.dll"})

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


def test_forced_cuda_without_a_usable_cuda_is_refused_with_a_reason(wc, monkeypatch) -> None:
    # Round 16: ABSTRACTVOICE_WHISPER_DEVICE=cuda used to be returned as is, so a host without a
    # working CUDA failed to load the model (no STT at all). Now: the CPU, refused, said why.
    from abstractvoice.compute.device import best_faster_whisper_device, resolve_faster_whisper_device

    _fake_ctranslate2(monkeypatch, cuda_devices=1)
    _simulate_windows(monkeypatch, loadable=set())
    monkeypatch.setenv("ABSTRACTVOICE_WHISPER_DEVICE", "cuda:0")
    choice = resolve_faster_whisper_device()
    assert (choice.device, choice.compute_type, choice.refused, choice.requested) == ("cpu", "int8", True, "cuda")
    assert choice.reason.startswith("ABSTRACTVOICE_WHISPER_DEVICE=cuda:0 refused:")
    assert "cublas64_12.dll" in choice.reason
    assert best_faster_whisper_device() == "cpu"


def test_forced_cuda_with_a_usable_cuda_is_honoured(wc, monkeypatch) -> None:
    from abstractvoice.compute.device import resolve_faster_whisper_device

    _fake_ctranslate2(monkeypatch, cuda_devices=1)
    _simulate_windows(monkeypatch, loadable={"cublas64_12.dll", "cudnn_ops64_9.dll"})
    monkeypatch.setenv("ABSTRACTVOICE_WHISPER_DEVICE", "cuda")
    choice = resolve_faster_whisper_device()
    assert (choice.device, choice.compute_type, choice.refused, choice.reason) == ("cuda", "int8_float16", False, None)


def test_macos_nothing_is_added_and_cublas_is_not_probed(wc, tmp_path, monkeypatch) -> None:
    from abstractvoice.compute.device import best_faster_whisper_device

    _fake_nvidia(tmp_path, monkeypatch)
    _fake_ctranslate2(monkeypatch, cuda_devices=1)
    monkeypatch.setattr(sys, "platform", "darwin")
    before = os.environ.get("PATH")

    assert best_faster_whisper_device() == "cuda"
    assert wc.prepare_windows_cuda_dlls() == []
    assert os.environ.get("PATH") == before
    assert wc._cublas12 is None


def _fake_nvidia_linux(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, files: dict) -> Path:
    """A fake `site-packages/nvidia` with the Linux layout (`<lib>/lib/<file>`)."""

    root = tmp_path / "site-packages" / "nvidia"
    for name, filenames in files.items():
        folder = root / name / "lib"
        folder.mkdir(parents=True)
        for filename in filenames:
            (folder / filename).write_bytes(b"")
    spec = importlib.util.spec_from_loader("nvidia", loader=None, is_package=True)
    spec.submodule_search_locations = [str(root)]
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name, *a, **k: spec if name == "nvidia" else real_find_spec(name, *a, **k)
    )
    return root


def _simulate_linux(monkeypatch: pytest.MonkeyPatch, loadable_names: set) -> list:
    """sys.platform = linux; ctypes.CDLL records every load. A path loads when the file exists
    (and, once loaded, its basename becomes loadable by name, as with RTLD_GLOBAL); a bare name
    loads when it is in `loadable_names` or was preloaded."""

    import ctypes

    loads: list = []
    preloaded: set = set()
    monkeypatch.setattr(sys, "platform", "linux")

    def fake_cdll(name, mode=0, *a, **k):
        loads.append((name, mode))
        if os.sep in str(name):
            if not Path(name).is_file():
                raise OSError(f"{name}: cannot open shared object file")
            preloaded.add(Path(name).name)
            return object()
        if name in loadable_names or name in preloaded:
            return object()
        raise OSError(f"{name}: cannot open shared object file: No such file or directory")

    monkeypatch.setattr(ctypes, "CDLL", fake_cdll)
    return loads


def test_linux_preloads_cuda12_runtime_and_cublas_from_the_nvidia_wheels_then_picks_cuda(
    wc, tmp_path, monkeypatch
) -> None:
    # Measured on a Linux + NVIDIA gpu install (backlog 0989): without the preload CTranslate2
    # fails at the first GPU transcription with "Library libcublas.so.12 is not found".
    import ctypes

    from abstractvoice.compute.device import best_faster_whisper_device

    root = _fake_nvidia_linux(
        tmp_path,
        monkeypatch,
        {
            "cuda_runtime": ["libcudart.so.12"],
            "cublas": ["libcublasLt.so.12", "libcublas.so.12"],
            "cudnn": ["libcudnn.so.9"],
        },
    )
    _fake_ctranslate2(monkeypatch, cuda_devices=1)
    loads = _simulate_linux(monkeypatch, loadable_names=set())
    before = os.environ.get("PATH")

    assert best_faster_whisper_device() == "cuda"
    preloads = [(n, m) for n, m in loads if os.sep in str(n)]
    assert [n for n, _ in preloads] == [
        str(root / "cuda_runtime" / "lib" / "libcudart.so.12"),
        str(root / "cublas" / "lib" / "libcublasLt.so.12"),
        str(root / "cublas" / "lib" / "libcublas.so.12"),
        str(root / "cudnn" / "lib" / "libcudnn.so.9"),
    ]
    assert all(m == ctypes.RTLD_GLOBAL for _, m in preloads)
    assert ("libcublas.so.12", 0) in loads
    assert os.environ.get("PATH") == before


def test_linux_without_cublas12_falls_back_to_cpu_with_a_warning(wc, tmp_path, monkeypatch, caplog) -> None:
    from abstractvoice.compute.device import best_faster_whisper_device

    _fake_nvidia_linux(tmp_path, monkeypatch, {"cu13": ["libcublas.so.13"]})
    _fake_ctranslate2(monkeypatch, cuda_devices=1)
    _simulate_linux(monkeypatch, loadable_names=set())

    with caplog.at_level("WARNING"):
        assert best_faster_whisper_device() == "cpu"
    assert "libcublas.so.12" in caplog.text


def test_linux_with_a_system_cublas12_picks_cuda_without_wheels(wc, monkeypatch) -> None:
    from abstractvoice.compute.device import best_faster_whisper_device

    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name, *a, **k: None if name == "nvidia" else real_find_spec(name, *a, **k)
    )
    _fake_ctranslate2(monkeypatch, cuda_devices=1)
    loads = _simulate_linux(monkeypatch, loadable_names={"libcublas.so.12", "libcudnn.so.9"})

    assert best_faster_whisper_device() == "cuda"
    assert [n for n, _ in loads] == ["libcublas.so.12", "libcudnn.so.9"]


def test_linux_without_cuda_stays_on_cpu_without_probing_cublas(wc, monkeypatch) -> None:
    from abstractvoice.compute.device import best_faster_whisper_device

    _fake_ctranslate2(monkeypatch, cuda_devices=0)
    _simulate_linux(monkeypatch, loadable_names={"libcublas.so.12"})

    assert best_faster_whisper_device() == "cpu"
    assert wc._cublas12 is None


def test_windows_without_nvidia_wheels_adds_nothing(wc, monkeypatch) -> None:
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name, *a, **k: None if name == "nvidia" else real_find_spec(name, *a, **k)
    )
    added = _simulate_windows(monkeypatch, loadable=set())
    assert wc.prepare_windows_cuda_dlls() == []
    assert added == []
