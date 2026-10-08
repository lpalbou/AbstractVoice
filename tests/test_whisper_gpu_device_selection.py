"""Where faster-whisper runs (round 16): CUDA when it really works, else the CPU WITH the reason.

CTranslate2, the platform and the CUDA libraries are simulated (a fake `ctranslate2` module, a
fake `ctypes.CDLL`), so these run on any machine, including the macOS CI. The real-hardware check
is the manual NVIDIA recipe in docs/installation.md ("Check Whisper on an NVIDIA GPU").
"""

from __future__ import annotations

import ctypes
import importlib.util
import sys
import types

import pytest

ALL_CUDA_TYPES = {"float32", "int8", "int8_float32", "int8_float16", "float16", "bfloat16", "int8_bfloat16"}


@pytest.fixture()
def env(monkeypatch: pytest.MonkeyPatch):
    from abstractvoice.compute import windows_cuda

    monkeypatch.setattr(windows_cuda, "_prepared", None)
    monkeypatch.setattr(windows_cuda, "_cublas12", None)
    monkeypatch.setattr(windows_cuda, "_cudnn9", None)
    monkeypatch.setattr(windows_cuda, "_handles", [])
    monkeypatch.delenv("ABSTRACTVOICE_WHISPER_DEVICE", raising=False)
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name, *a, **k: None if name == "nvidia" else real_find_spec(name, *a, **k)
    )
    return monkeypatch


def _ct2(monkeypatch, devices: int, cuda_types=ALL_CUDA_TYPES, count_raises: bool = False) -> None:
    module = types.ModuleType("ctranslate2")

    def count():
        if count_raises:
            raise RuntimeError("This CTranslate2 package was not compiled with CUDA support")
        return devices

    def supported(device, index=0):
        if device == "cpu":
            return {"int8", "int8_float32", "float32"}
        if devices > 0 and not count_raises:
            return set(cuda_types)
        raise ValueError("This CTranslate2 package was not compiled with CUDA support")

    module.get_cuda_device_count = count  # type: ignore[attr-defined]
    module.get_supported_compute_types = supported  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ctranslate2", module)


def _linux(monkeypatch, loadable: set) -> None:
    monkeypatch.setattr(sys, "platform", "linux")

    def fake_cdll(name, mode=0, *a, **k):
        if name in loadable:
            return object()
        raise OSError(f"{name}: cannot open shared object file")

    monkeypatch.setattr(ctypes, "CDLL", fake_cdll)


CUDA_LIBS = {"libcublas.so.12", "libcudnn.so.9"}


def test_one_gpu_with_cublas_and_cudnn_picks_cuda_and_int8_float16(env) -> None:
    from abstractvoice.compute.device import resolve_faster_whisper_device

    _ct2(env, devices=1)
    _linux(env, CUDA_LIBS)
    choice = resolve_faster_whisper_device()
    assert (choice.device, choice.compute_type, choice.reason, choice.refused) == ("cuda", "int8_float16", None, False)


def test_one_gpu_without_cublas_runs_on_the_cpu_and_says_why(env) -> None:
    from abstractvoice.compute.device import resolve_faster_whisper_device

    _ct2(env, devices=1)
    _linux(env, {"libcudnn.so.9"})
    choice = resolve_faster_whisper_device()
    assert (choice.device, choice.compute_type, choice.refused) == ("cpu", "int8", False)
    assert "libcublas.so.12" in choice.reason and 'abstractvoice[gpu]' in choice.reason


def test_one_gpu_without_cudnn_runs_on_the_cpu_and_says_why(env) -> None:
    # The Whisper encoder's convolutions need cuDNN 9 on the GPU: without it the first GPU
    # transcription failed ("Could not load library libcudnn_ops.so.9").
    from abstractvoice.compute.device import resolve_faster_whisper_device

    _ct2(env, devices=1)
    _linux(env, {"libcublas.so.12"})
    choice = resolve_faster_whisper_device()
    assert choice.device == "cpu"
    assert "libcudnn.so.9" in choice.reason and "cuDNN 9" in choice.reason


def test_no_gpu_runs_on_the_cpu_with_a_reason_and_probes_no_cuda_library(env) -> None:
    from abstractvoice.compute import windows_cuda
    from abstractvoice.compute.device import resolve_faster_whisper_device

    _ct2(env, devices=0)
    _linux(env, CUDA_LIBS)
    choice = resolve_faster_whisper_device()
    assert (choice.device, choice.compute_type) == ("cpu", "int8")
    assert choice.reason == "CTranslate2 sees no CUDA GPU"
    assert windows_cuda._cublas12 is None and windows_cuda._cudnn9 is None


def test_a_ctranslate2_build_without_cuda_is_a_reason_not_a_crash(env) -> None:
    from abstractvoice.compute.device import resolve_faster_whisper_device

    _ct2(env, devices=0, count_raises=True)
    _linux(env, CUDA_LIBS)
    choice = resolve_faster_whisper_device()
    assert choice.device == "cpu" and "no CUDA support" in choice.reason


def test_macos_reason_names_the_apple_gpu_engine(env) -> None:
    from abstractvoice.compute.device import resolve_faster_whisper_device

    _ct2(env, devices=0)
    env.setattr(sys, "platform", "darwin")
    choice = resolve_faster_whisper_device()
    assert choice.device == "cpu"
    assert "no Apple GPU backend" in choice.reason and "mlx-whisper" in choice.reason


@pytest.mark.parametrize(
    "cuda_types,expected",
    [
        ({"float32", "int8", "int8_float32", "float16", "int8_float16"}, "int8_float16"),
        ({"float32", "float16"}, "float16"),  # no int8 kernels (compute capability < 6.1)
        ({"float32", "int8", "int8_float32"}, "int8"),  # no float16 (compute capability < 5.3)
        ({"float32"}, "float32"),
    ],
)
def test_compute_type_follows_what_the_gpu_supports(env, cuda_types, expected) -> None:
    from abstractvoice.compute.device import resolve_faster_whisper_device

    _ct2(env, devices=1, cuda_types=cuda_types)
    _linux(env, CUDA_LIBS)
    assert resolve_faster_whisper_device().compute_type == expected


def test_forced_cuda_without_cuda_is_refused_with_a_sentence(env) -> None:
    from abstractvoice.compute.device import best_faster_whisper_device, resolve_faster_whisper_device

    _ct2(env, devices=0)
    _linux(env, CUDA_LIBS)
    env.setenv("ABSTRACTVOICE_WHISPER_DEVICE", "cuda")
    choice = resolve_faster_whisper_device()
    assert (choice.device, choice.compute_type, choice.refused, choice.requested) == ("cpu", "int8", True, "cuda")
    assert choice.reason == (
        "ABSTRACTVOICE_WHISPER_DEVICE=cuda refused: CTranslate2 sees no CUDA GPU; running on the processor"
    )
    assert best_faster_whisper_device() == "cpu"


def test_forced_cpu_is_honoured_even_with_a_working_gpu(env) -> None:
    from abstractvoice.compute.device import resolve_faster_whisper_device

    _ct2(env, devices=1)
    _linux(env, CUDA_LIBS)
    env.setenv("ABSTRACTVOICE_WHISPER_DEVICE", "cpu")
    choice = resolve_faster_whisper_device()
    assert (choice.device, choice.refused) == ("cpu", False)
    assert "ABSTRACTVOICE_WHISPER_DEVICE=cpu" in choice.reason


def test_an_unknown_forced_device_is_refused(env) -> None:
    from abstractvoice.compute.device import resolve_faster_whisper_device

    _ct2(env, devices=1)
    _linux(env, CUDA_LIBS)
    env.setenv("ABSTRACTVOICE_WHISPER_DEVICE", "mps")
    choice = resolve_faster_whisper_device()
    assert (choice.device, choice.refused) == ("cpu", True)
    assert "not a faster-whisper device" in choice.reason


# --- the adapter: fallback to the CPU with the reason recorded -----------------------------


class _Seg:
    def __init__(self, text: str) -> None:
        self.text = text


class _Info:
    language = "en"
    language_probability = 1.0


def _fake_whisper_model(fail_load_on_cuda: bool = False, fail_run_on_cuda: bool = False):
    built = []

    class FakeWhisperModel:
        def __init__(self, model, device="cpu", compute_type="int8", **kwargs):
            built.append((model, device, compute_type))
            if device == "cuda" and fail_load_on_cuda:
                raise RuntimeError("CUDA failed with error out of memory")
            self.device = device

        def transcribe(self, audio, **kwargs):
            def gen():
                if self.device == "cuda" and fail_run_on_cuda:
                    raise RuntimeError("Unable to load any of {libcudnn_ops.so.9.1.0, libcudnn_ops.so.9}")
                yield _Seg(f"hello from {self.device}")

            return gen(), _Info()

    return FakeWhisperModel, built


def _adapter(env, fake_model, devices=1, libs=CUDA_LIBS):
    from abstractvoice.adapters import stt_faster_whisper

    _ct2(env, devices=devices)
    _linux(env, libs)
    module = types.ModuleType("faster_whisper")
    module.WhisperModel = fake_model  # type: ignore[attr-defined]
    env.setitem(sys.modules, "faster_whisper", module)
    return stt_faster_whisper.FasterWhisperAdapter(model_size="large-v3", device="auto", compute_type="auto")


def test_adapter_on_a_working_gpu_loads_cuda_with_the_capability_compute_type(env) -> None:
    fake, built = _fake_whisper_model()
    adapter = _adapter(env, fake)
    assert built == [("large-v3", "cuda", "int8_float16")]
    assert adapter.execution_device() == {
        "device": "cuda", "compute_type": "int8_float16", "reason": None, "refused": False
    }
    assert "device_reason" not in adapter.get_info()


def test_adapter_falls_back_to_the_cpu_when_cuda_fails_to_load_and_records_why(env) -> None:
    fake, built = _fake_whisper_model(fail_load_on_cuda=True)
    adapter = _adapter(env, fake)
    assert adapter.is_available()
    assert built == [("large-v3", "cuda", "int8_float16"), ("large-v3", "cpu", "int8")]
    info = adapter.get_info()
    assert info["device"] == "cpu" and info["compute_type"] == "int8"
    assert info["device_reason"] == (
        "loading on CUDA failed (CUDA failed with error out of memory); running on the processor"
    )


def test_adapter_falls_back_to_the_cpu_when_the_first_gpu_run_fails_and_retries(env) -> None:
    import numpy as np

    fake, built = _fake_whisper_model(fail_run_on_cuda=True)
    adapter = _adapter(env, fake)
    text = adapter.transcribe_from_array(np.zeros(16000, dtype=np.float32), 16000, language="en")
    assert text == "hello from cpu"
    assert built[-1] == ("large-v3", "cpu", "int8")
    reason = adapter.execution_device()["reason"]
    assert reason.startswith("transcribing on CUDA failed (Unable to load any of {libcudnn_ops.so.9")
    assert reason.endswith("; running on the processor")


def test_adapter_without_a_gpu_records_the_reason(env) -> None:
    fake, built = _fake_whisper_model()
    adapter = _adapter(env, fake, devices=0)
    assert built == [("large-v3", "cpu", "int8")]
    assert adapter.get_info()["device_reason"] == "CTranslate2 sees no CUDA GPU"


def test_adapter_forced_cuda_without_cuda_is_refused_not_a_failed_load(env) -> None:
    fake, built = _fake_whisper_model()
    env.setenv("ABSTRACTVOICE_WHISPER_DEVICE", "cuda")
    adapter = _adapter(env, fake, devices=0)
    assert adapter.is_available() and built == [("large-v3", "cpu", "int8")]
    info = adapter.get_info()
    assert info["device_refused"] is True
    assert info["device_reason"].startswith("ABSTRACTVOICE_WHISPER_DEVICE=cuda refused:")


def test_large_v3_turbo_is_offered_and_large_v3_stays_listed_first_of_the_large_models() -> None:
    from abstractvoice.adapters.stt_faster_whisper import FasterWhisperAdapter

    ids = FasterWhisperAdapter.selectable_model_ids()
    assert "large-v3-turbo" in ids and "turbo" in ids
    assert ids.index("large-v3") < ids.index("large-v3-turbo")
