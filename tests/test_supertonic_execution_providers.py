"""Supertonic runs on an NVIDIA GPU when ONNX Runtime has the CUDA provider, on the CPU
otherwise, and reports the providers its sessions ACTUALLY run on (ONNX Runtime falls
back to the CPU, with a log line only, when the CUDA provider cannot load).

ONNX Runtime is faked: these run anywhere, without onnxruntime-gpu or a GPU."""
from __future__ import annotations

import importlib.machinery
import json
import sys
import types
from types import SimpleNamespace

import pytest

CUDA = "CUDAExecutionProvider"
CPU = "CPUExecutionProvider"
COREML = "CoreMLExecutionProvider"


def _fake_ort(available, *, loadable=None, preload_prints=""):
    """An `onnxruntime` stand-in. `loadable`: the providers a session really gets (default: all available)."""
    ort = types.ModuleType("onnxruntime")
    ort.__spec__ = importlib.machinery.ModuleSpec("onnxruntime", None)  # visible to find_spec
    ort.sessions = []
    ort.preload_calls = 0
    loads = set(available if loadable is None else loadable)

    class InferenceSession:
        def __init__(self, path, sess_options=None, providers=None):
            self.path = str(path)
            self.requested = list(providers or [])
            self._providers = [p for p in self.requested if p in loads] or [CPU]
            ort.sessions.append(self)

        def get_providers(self):
            return list(self._providers)

    def preload_dlls(cuda=True, cudnn=True, msvc=True, directory=None):
        ort.preload_calls += 1
        if preload_prints:
            print(preload_prints)

    ort.get_available_providers = lambda: list(available)
    ort.InferenceSession = InferenceSession
    ort.SessionOptions = lambda: SimpleNamespace()
    ort.GraphOptimizationLevel = SimpleNamespace(ORT_ENABLE_ALL=99)
    ort.ExecutionMode = SimpleNamespace(ORT_SEQUENTIAL=0)
    ort.preload_dlls = preload_dlls
    return ort


@pytest.fixture
def use_ort(monkeypatch):
    monkeypatch.delenv("ABSTRACTVOICE_SUPERTONIC_ONNX_PROVIDERS", raising=False)

    def install(ort):
        monkeypatch.setitem(sys.modules, "onnxruntime", ort)
        return ort

    return install


def _cache(tmp_path):
    """A cache directory that passes `is_supertonic_cached` and parses at load."""
    from abstractvoice.supertonic import runtime as rt

    for rel in rt._REQUIRED_FILES:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    cfg = {"ae": {"sample_rate": 44100, "base_chunk_size": 512}, "ttl": {"chunk_compress_factor": 6, "latent_dim": 24}}
    (tmp_path / "onnx" / "tts.json").write_text(json.dumps(cfg))
    (tmp_path / "onnx" / "unicode_indexer.json").write_text(json.dumps([0] * 128))
    return tmp_path


def _runtime(tmp_path, **kwargs):
    from abstractvoice.supertonic.runtime import SupertonicRuntime

    return SupertonicRuntime(cache_dir=_cache(tmp_path), allow_downloads=False, **kwargs)


def test_cuda_available_puts_the_cuda_provider_first(use_ort, tmp_path) -> None:
    ort = use_ort(_fake_ort(["TensorrtExecutionProvider", CUDA, CPU]))
    runtime = _runtime(tmp_path)

    assert runtime._resolve_providers() == [CUDA, CPU]
    assert runtime.execution_providers is None  # nothing loaded yet
    runtime._ensure_loaded()

    assert ort.preload_calls == 1  # the pip-installed CUDA libraries are loaded first
    assert len(ort.sessions) == 4
    assert all(s.requested == [CUDA, CPU] for s in ort.sessions)
    assert runtime.execution_providers == [CUDA, CPU]
    info = runtime.execution_device()
    assert info["device"] == "cuda"
    assert info["reason"] == runtime.CUDA_REASON
    assert info["execution_providers"] == [CUDA, CPU]


def test_cuda_absent_runs_on_the_cpu_only(use_ort, tmp_path, monkeypatch) -> None:
    from abstractvoice.supertonic import runtime as rt

    ort = use_ort(_fake_ort(["AzureExecutionProvider", CPU]))
    runtime = _runtime(tmp_path)

    assert runtime._resolve_providers() == [CPU]
    runtime._ensure_loaded()

    assert ort.preload_calls == 0  # no CUDA requested: nothing to preload
    assert all(s.requested == [CPU] for s in ort.sessions)
    assert runtime.execution_providers == [CPU]
    monkeypatch.setattr(rt.sys, "platform", "linux")
    monkeypatch.setattr(rt, "onnxruntime_gpu_installed", lambda: False)
    linux = runtime.execution_device()
    assert linux["device"] == "cpu" and linux["reason"] == runtime.NO_CUDA_REASON
    # The gpu profile installed onnxruntime-gpu, then the CPU onnxruntime (piper-tts, faster-whisper)
    # overwrote its files: say so, with the fix.
    monkeypatch.setattr(rt, "onnxruntime_gpu_installed", lambda: True)
    overwritten = runtime.execution_device()["reason"]
    assert overwritten == runtime.CUDA_OVERWRITTEN_REASON
    assert "pip install --force-reinstall --no-deps onnxruntime-gpu" in overwritten
    monkeypatch.setattr(rt.sys, "platform", "darwin")
    assert runtime.execution_device()["reason"] == runtime.CPU_REASON


def test_cuda_provider_that_fails_to_load_is_reported_as_cpu(use_ort, tmp_path, monkeypatch, caplog) -> None:
    # onnxruntime-gpu without its CUDA libraries: listed as available, but each session gets the CPU.
    ort = use_ort(_fake_ort([CUDA, CPU], loadable=[CPU], preload_prints="Failed to load libcurand.so.10"))
    runtime = _runtime(tmp_path)
    caplog.set_level("WARNING", logger="abstractvoice.compute.onnx_cuda")

    runtime._ensure_loaded()

    assert ort.sessions[0].requested == [CUDA, CPU]
    # The first session told the truth: the other three do not retry (one load error, not four).
    assert all(s.requested == [CPU] for s in ort.sessions[1:])
    assert runtime.execution_providers == [CPU]
    info = runtime.execution_device()
    assert info["device"] == "cpu"
    assert info["reason"] == runtime.CUDA_FAILED_REASON
    assert info["providers"] == [CUDA, CPU]  # what was requested ...
    assert info["execution_providers"] == [CPU]  # ... and what runs
    # preload_dlls prints; that goes to the log, never to stdout.
    assert "libcurand.so.10" in caplog.text


def test_preload_output_never_reaches_stdout(use_ort, tmp_path, capsys) -> None:
    use_ort(_fake_ort([CUDA, CPU], preload_prints="Failed to load libcudnn.so.9"))
    _runtime(tmp_path)._ensure_loaded()
    assert "libcudnn" not in capsys.readouterr().out


def test_explicit_providers_are_respected(use_ort, tmp_path) -> None:
    ort = use_ort(_fake_ort([CUDA, COREML, CPU]))
    runtime = _runtime(tmp_path, providers=[CPU])

    assert runtime._resolve_providers() == [CPU]
    runtime._ensure_loaded()
    assert ort.preload_calls == 0
    assert all(s.requested == [CPU] for s in ort.sessions)
    info = runtime.execution_device()
    assert info["device"] == "cpu" and info["reason"] == runtime.CPU_REQUESTED_REASON

    # Unavailable providers are dropped, the order of the rest is kept.
    assert _runtime(tmp_path, providers=["TensorrtExecutionProvider", COREML, CPU])._resolve_providers() == [COREML, CPU]


def test_env_override_still_wins_over_the_cuda_default(use_ort, tmp_path, monkeypatch) -> None:
    use_ort(_fake_ort([CUDA, CPU]))
    monkeypatch.setenv("ABSTRACTVOICE_SUPERTONIC_ONNX_PROVIDERS", "CPUExecutionProvider")
    assert _runtime(tmp_path)._resolve_providers() == [CPU]
    # An explicit argument beats the environment.
    assert _runtime(tmp_path, providers=[CUDA])._resolve_providers() == [CUDA]


def test_adapter_info_reports_the_providers_in_use(use_ort, tmp_path) -> None:
    from abstractvoice.adapters.tts_supertonic import SupertonicTTSAdapter

    use_ort(_fake_ort([CUDA, CPU]))
    cache = _cache(tmp_path)
    adapter = SupertonicTTSAdapter(cache_dir=str(cache), allow_downloads=False, auto_load=False)

    info = adapter.get_info()
    assert info["device"] == "cuda"
    assert info["execution_providers"] == [CUDA, CPU]
    assert adapter.execution_device()["reason"] == adapter._runtime.CUDA_REASON


def test_adapter_without_onnxruntime_reports_no_providers(tmp_path) -> None:
    from abstractvoice.adapters.tts_supertonic import SupertonicTTSAdapter

    adapter = SupertonicTTSAdapter(cache_dir=str(tmp_path), allow_downloads=False, auto_load=False)
    adapter._onnx_available = False
    assert adapter.execution_device() == {
        "device": "cpu",
        "reason": "ONNX Runtime is not installed",
        "execution_providers": None,
    }
