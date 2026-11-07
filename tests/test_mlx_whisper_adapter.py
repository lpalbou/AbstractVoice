"""mlx-whisper: Whisper on the Apple GPU behind the same STTAdapter interface (round 16).

`mlx`, `mlx_whisper` and the Hugging Face download are simulated, so these run anywhere. The real
engine is measured by the optional test at the bottom (Apple Silicon, mlx-whisper installed, the
model cached) and by the before/after table in docs/engines.md.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import threading
import types

import numpy as np
import pytest


@pytest.fixture()
def fake_mlx(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Fake mlx + mlx_whisper that FAIL off the thread that loaded the model, like real MLX."""

    from abstractvoice.adapters import stt_mlx_whisper as mod

    calls: list = []
    state = {"loader_thread": None}

    mx = types.ModuleType("mlx.core")
    mx.float16 = "float16"  # type: ignore[attr-defined]
    mx.clear_cache = lambda: calls.append(("clear_cache", threading.get_ident()))  # type: ignore[attr-defined]
    mlx_pkg = types.ModuleType("mlx")
    mlx_pkg.core = mx  # type: ignore[attr-defined]

    class ModelHolder:
        model = None
        model_path = None

        @classmethod
        def get_model(cls, path, dtype):
            if cls.model is None or path != cls.model_path:
                calls.append(("load", path, threading.get_ident()))
                cls.model, cls.model_path = object(), path
                state["loader_thread"] = threading.get_ident()
            return cls.model

    transcribe_mod = types.ModuleType("mlx_whisper.transcribe")
    transcribe_mod.ModelHolder = ModelHolder  # type: ignore[attr-defined]

    def transcribe(audio, *, path_or_hf_repo, language=None, **kwargs):
        if threading.get_ident() != state["loader_thread"]:
            raise RuntimeError("There is no Stream(gpu, 1) in current thread.")
        ModelHolder.get_model(path_or_hf_repo, "float16")
        calls.append(("transcribe", path_or_hf_repo, language, len(audio), threading.get_ident()))
        return {"text": f" heard {len(audio)} samples "}

    mw = types.ModuleType("mlx_whisper")
    mw.transcribe = transcribe  # type: ignore[attr-defined]
    mw.transcribe_module = transcribe_mod  # type: ignore[attr-defined]
    for name, module in (("mlx", mlx_pkg), ("mlx.core", mx), ("mlx_whisper", mw), ("mlx_whisper.transcribe", transcribe_mod)):
        monkeypatch.setitem(sys.modules, name, module)

    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec",
        lambda name, *a, **k: object() if name == "mlx_whisper" else real_find_spec(name, *a, **k),
    )
    monkeypatch.setattr(mod, "apple_gpu_available", lambda: True)

    downloads: list = []

    def fake_snapshot(repo, local_files_only=False, **kwargs):
        downloads.append((repo, local_files_only))
        if local_files_only and "missing" in repo:
            raise FileNotFoundError(f"{repo} not in the cache")
        folder = tmp_path / repo.replace("/", "--")
        folder.mkdir(parents=True, exist_ok=True)
        return str(folder)

    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = fake_snapshot  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    return types.SimpleNamespace(mod=mod, calls=calls, downloads=downloads, holder=ModelHolder, tmp=tmp_path)


def test_model_ids_are_faster_whispers_plus_turbo_with_their_mlx_repos() -> None:
    from abstractvoice.adapters.stt_faster_whisper import FasterWhisperAdapter
    from abstractvoice.adapters.stt_mlx_whisper import MLXWhisperAdapter

    assert MLXWhisperAdapter.DEFAULT_MODEL == "large-v3"
    assert MLXWhisperAdapter.resolve_repo("large-v3") == "mlx-community/whisper-large-v3-mlx"
    assert MLXWhisperAdapter.resolve_repo("large") == "mlx-community/whisper-large-v3-mlx"
    assert MLXWhisperAdapter.resolve_repo("large-v3-turbo") == "mlx-community/whisper-large-v3-turbo"
    assert MLXWhisperAdapter.resolve_repo("turbo") == "mlx-community/whisper-large-v3-turbo"
    assert MLXWhisperAdapter.resolve_repo("someone/custom-mlx") == "someone/custom-mlx"
    # A route can move between the two Whisper engines without changing its model id.
    assert set(FasterWhisperAdapter.selectable_model_ids()) == set(MLXWhisperAdapter.selectable_model_ids())


def test_load_and_every_transcription_run_on_one_mlx_thread_whatever_the_caller(fake_mlx) -> None:
    adapter = fake_mlx.mod.MLXWhisperAdapter("large-v3", allow_downloads=False)
    assert adapter.is_available()
    out: dict = {}

    def call(tag):
        out[tag] = adapter.transcribe_from_array(np.zeros(8000, dtype=np.float32), 8000, language="en")

    threads = [threading.Thread(target=call, args=(t,)) for t in ("a", "b")]
    for t in threads:
        t.start()
        t.join()
    call("main")
    assert out == {k: "heard 16000 samples" for k in ("a", "b", "main")}  # resampled to 16 kHz
    threads_used = {c[-1] for c in fake_mlx.calls}
    assert len(threads_used) == 1 and threading.get_ident() not in threads_used
    assert fake_mlx.downloads == [("mlx-community/whisper-large-v3-mlx", True)]


def test_a_new_model_id_switches_the_weights_on_the_next_call(fake_mlx) -> None:
    adapter = fake_mlx.mod.MLXWhisperAdapter("large-v3")
    adapter.model_id = "large-v3-turbo"  # what the AbstractCore plugin does per request
    adapter.transcribe_from_array(np.zeros(16000, dtype=np.float32), 16000)
    loads = [c[1] for c in fake_mlx.calls if c[0] == "load"]
    assert [os.path.basename(p) for p in loads] == [
        "mlx-community--whisper-large-v3-mlx",
        "mlx-community--whisper-large-v3-turbo",
    ]


def test_a_model_that_is_not_downloaded_is_unavailable_with_a_reason(fake_mlx) -> None:
    adapter = fake_mlx.mod.MLXWhisperAdapter("someone/missing-mlx", allow_downloads=False)
    assert not adapter.is_available()
    assert adapter.get_unavailable_reason() == "mlx-whisper model 'someone/missing-mlx' (someone/missing-mlx) is not downloaded"


def test_off_apple_silicon_the_engine_is_unavailable_and_says_so(monkeypatch) -> None:
    from abstractvoice.adapters import stt_mlx_whisper as mod

    monkeypatch.setattr(mod, "apple_gpu_available", lambda: False)
    adapter = mod.MLXWhisperAdapter("large-v3")
    assert not adapter.is_available()
    assert "Apple Silicon only" in adapter.get_unavailable_reason()


def test_reports_the_apple_gpu_as_its_device(fake_mlx) -> None:
    adapter = fake_mlx.mod.MLXWhisperAdapter("large-v3")
    info = adapter.get_info()
    assert (info["engine_id"], info["device"], info["model_repo"]) == (
        "mlx-whisper", "metal", "mlx-community/whisper-large-v3-mlx"
    )
    assert adapter.execution_device() == {"device": "metal", "reason": "Apple GPU (MLX)"}


def test_wav_bytes_and_files_decode_to_16k_mono(fake_mlx, tmp_path) -> None:
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(48000)
        w.writeframes(np.zeros((48000, 2), dtype=np.int16).tobytes())
    adapter = fake_mlx.mod.MLXWhisperAdapter("large-v3")
    assert adapter.transcribe_from_bytes(buf.getvalue()) == "heard 16000 samples"
    path = tmp_path / "clip.wav"
    path.write_bytes(buf.getvalue())
    assert adapter.transcribe(str(path)) == "heard 16000 samples"


def test_engine_runtime_and_plugin_list_mlx_whisper_as_a_local_stt_engine() -> None:
    from abstractvoice.engine_runtime import engine_runtime_status, known_engines
    from abstractvoice.integrations import abstractcore_plugin as plugin

    assert "mlx-whisper" in known_engines("stt")
    status = engine_runtime_status("mlx_whisper")
    assert (status.engine, status.extra, status.required_modules) == ("mlx-whisper", "stt-mlx", ("mlx_whisper",))
    assert plugin._known_stt_provider_ids() == [
        "openai", "openai-compatible", "faster-whisper", "mlx-whisper", "transformers-asr"
    ]
    assert "large-v3-turbo" in plugin._stt_model_ids_for_provider("mlx-whisper")
    assert "large-v3-turbo" in plugin._stt_model_ids_for_provider("faster-whisper")


def test_voice_manager_builds_the_mlx_adapter_for_the_mlx_whisper_engine(fake_mlx) -> None:
    from abstractvoice.vm.stt_mixin import SttMixin

    class VM(SttMixin):
        debug_mode = False
        allow_downloads = False
        language = "fr"
        whisper_model = "large-v3"
        stt_model = None
        stt_adapter = None
        _stt_engine_preference = "mlx-whisper"

    adapter = VM()._get_stt_adapter()
    assert adapter.engine_id == "mlx-whisper" and adapter.model_id == "large-v3"
    # Round 18: the manager's `language` is the TTS voice's; the STT adapter starts on AUTO
    # (None) so a transcription without a language hint is detected, never forced to "fr".
    assert adapter._current_language is None


_REAL = (
    sys.platform == "darwin"
    and os.uname().machine == "arm64"
    and importlib.util.find_spec("mlx_whisper") is not None
)


@pytest.mark.model_download
@pytest.mark.skipif(not _REAL, reason="needs Apple Silicon + mlx-whisper + the cached turbo model")
def test_real_mlx_whisper_transcribes_speech_on_the_apple_gpu() -> None:  # pragma: no cover - hardware
    import math

    from abstractvoice.adapters.stt_mlx_whisper import MLXWhisperAdapter

    adapter = MLXWhisperAdapter("large-v3-turbo", allow_downloads=False)
    if not adapter.is_available():
        pytest.skip(adapter.get_unavailable_reason())
    tone = np.array([0.1 * math.sin(2 * math.pi * 440 * i / 16000) for i in range(16000)], dtype=np.float32)
    assert isinstance(adapter.transcribe_from_array(tone, 16000, language="en"), str)
