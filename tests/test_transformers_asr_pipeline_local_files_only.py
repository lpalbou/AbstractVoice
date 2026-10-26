"""``transformers-asr`` loads Whisper-class models itself and passes ``local_files_only``.

Torch-free. A stub ``transformers`` reproduces what Transformers 5.x ``pipeline()``
does with a model *string*: it merges its own hub kwargs (which carry
``local_files_only``) with ``model_kwargs``, so the historical call
``pipeline(model="<id>", model_kwargs={"local_files_only": ...})`` raises the same
``TypeError`` here as on a real install. The adapter must load config, model,
tokenizer and feature extractor with ``local_files_only`` on every call, pick the
model class the way the pipeline does (mapping membership), and hand the objects
to ``pipeline()``.
"""
from __future__ import annotations

import sys
import types

import pytest


class _Calls:
    def __init__(self) -> None:
        self.from_pretrained: list[tuple[str, dict]] = []
        self.pipeline_kwargs: dict | None = None
        self.pipeline_model = None


class _WhisperConfig:
    model_type = "whisper"


class _Wav2Vec2Config:
    model_type = "wav2vec2"


class _UnknownConfig:
    model_type = "mystery"


def _install_stub_transformers(monkeypatch, calls: _Calls, *, version: str, config_cls) -> types.ModuleType:
    stub = types.ModuleType("transformers")
    stub.__version__ = version

    class _Param:
        device = "cpu"
        dtype = "torch.float32"

    class _Model:
        def parameters(self):
            yield _Param()

    class _Tokenizer:
        pass

    class _FeatureExtractor:
        sampling_rate = 16000

    def _auto(name, result, mapping=None):
        class _Auto:
            _model_mapping = tuple(mapping or ())

            @staticmethod
            def from_pretrained(model_id, **kwargs):
                calls.from_pretrained.append((name, dict(kwargs)))
                return result()

        _Auto.__name__ = name
        return _Auto

    stub.AutoConfig = _auto("AutoConfig", config_cls)
    stub.AutoModelForCTC = _auto("AutoModelForCTC", _Model, mapping=[_Wav2Vec2Config])
    stub.AutoModelForSpeechSeq2Seq = _auto("AutoModelForSpeechSeq2Seq", _Model, mapping=[_WhisperConfig])
    stub.AutoTokenizer = _auto("AutoTokenizer", _Tokenizer)
    stub.AutoFeatureExtractor = _auto("AutoFeatureExtractor", _FeatureExtractor)

    pipelines = types.ModuleType("transformers.pipelines")
    pipelines.SUPPORTED_TASKS = {
        "automatic-speech-recognition": {"pt": (stub.AutoModelForCTC, stub.AutoModelForSpeechSeq2Seq)}
    }

    def pipeline(task, model=None, **kwargs):
        calls.pipeline_model = model
        calls.pipeline_kwargs = dict(kwargs)
        if isinstance(model, str):
            # Transformers >= 5.0 behaviour for a model id string.
            hub_kwargs = {"local_files_only": kwargs.get("local_files_only", False)}
            model_kwargs = kwargs.get("model_kwargs") or {}
            stub.AutoConfig.from_pretrained(model, **hub_kwargs, **model_kwargs)  # TypeError on collision
        return types.SimpleNamespace(model=model, feature_extractor=_FeatureExtractor)

    utils = types.ModuleType("transformers.utils")
    utils.logging = types.SimpleNamespace(disable_progress_bar=lambda: None)

    stub.pipeline = pipeline
    stub.pipelines = pipelines
    stub.utils = utils
    monkeypatch.setitem(sys.modules, "transformers", stub)
    monkeypatch.setitem(sys.modules, "transformers.pipelines", pipelines)
    monkeypatch.setitem(sys.modules, "transformers.utils", utils)
    return stub


def _fresh_adapter(calls: _Calls, **kwargs):
    """Build the adapter, then forget whatever its best-effort eager load recorded.

    The constructor tries ``_ensure_loaded()`` and swallows failures; the tests
    below exercise one explicit ``_load_backend`` call and nothing else.
    """
    from abstractvoice.adapters.stt_transformers_asr import TransformersASRAdapter

    adapter = TransformersASRAdapter(**kwargs)
    adapter._pipeline = None
    calls.from_pretrained.clear()
    calls.pipeline_kwargs = None
    calls.pipeline_model = None
    return adapter


@pytest.mark.parametrize("local_only", [True, False])
@pytest.mark.parametrize(
    ("version", "dtype_key"),
    [("5.20.2", "dtype"), ("5.4.0", "dtype"), ("4.57.3", "dtype"), ("4.55.4", "torch_dtype")],
)
def test_whisper_loads_explicitly_with_local_files_only(monkeypatch, local_only: bool, version: str, dtype_key: str) -> None:
    calls = _Calls()
    stub = _install_stub_transformers(monkeypatch, calls, version=version, config_cls=_WhisperConfig)
    adapter = _fresh_adapter(calls, model_id="openai/whisper-large-v3-turbo", allow_downloads=not local_only)

    adapter._load_backend(pipeline=stub.pipeline, torch_device="cpu", torch_dtype=None, local_only=local_only)

    assert adapter._pipeline is not None
    assert not isinstance(calls.pipeline_model, str), "pipeline() must receive a loaded model, not a model id"
    assert "model_kwargs" not in calls.pipeline_kwargs and "local_files_only" not in calls.pipeline_kwargs
    assert "tokenizer" in calls.pipeline_kwargs and "feature_extractor" in calls.pipeline_kwargs
    loaders = [name for name, _ in calls.from_pretrained]
    assert loaders == ["AutoConfig", "AutoModelForSpeechSeq2Seq", "AutoTokenizer", "AutoFeatureExtractor"]
    for name, kwargs in calls.from_pretrained:
        assert kwargs.get("local_files_only") is local_only, f"{name} lost local_files_only={local_only}"
        assert kwargs.get("trust_remote_code") is False, f"{name} lost trust_remote_code"
    model_kwargs = dict(calls.from_pretrained[1][1])
    assert model_kwargs[dtype_key] == "auto"
    assert ({"dtype", "torch_dtype"} - {dtype_key}).isdisjoint(model_kwargs)


def test_ctc_models_dispatch_like_the_pipeline(monkeypatch) -> None:
    calls = _Calls()
    stub = _install_stub_transformers(monkeypatch, calls, version="5.20.2", config_cls=_Wav2Vec2Config)
    adapter = _fresh_adapter(calls, model_id="facebook/wav2vec2-base-960h", allow_downloads=False)

    adapter._load_backend(pipeline=stub.pipeline, torch_device="cpu", torch_dtype=None, local_only=True)

    assert [name for name, _ in calls.from_pretrained][1] == "AutoModelForCTC"


def test_unsupported_model_type_is_named_before_any_weight_load(monkeypatch) -> None:
    calls = _Calls()
    stub = _install_stub_transformers(monkeypatch, calls, version="5.20.2", config_cls=_UnknownConfig)
    adapter = _fresh_adapter(calls, model_id="someone/not-an-asr-model", allow_downloads=False)

    with pytest.raises(RuntimeError, match="mystery"):
        adapter._load_backend(pipeline=stub.pipeline, torch_device="cpu", torch_dtype=None, local_only=True)

    assert [name for name, _ in calls.from_pretrained] == ["AutoConfig"]
