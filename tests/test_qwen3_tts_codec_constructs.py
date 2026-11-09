"""The vendored Qwen3-TTS speech tokenizer constructs on the installed Transformers (no weights).

Transformers 5.21 split `MimiModel` into encoder and decoder sub-models and turned `upsample` and
`decoder_transformer` into read-only properties; the tokenizer's encoder, a `MimiModel` subclass,
then failed in `__init__` and no Qwen3-TTS model could load on the PyTorch backend. This test
builds the tokenizer from the 0.6B checkpoint's configuration (a copy under tests/data, no download)
so the CI job that runs against the newest Transformers goes red on that kind of drift.
"""
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

CONFIG = Path(__file__).parent / "data" / "qwen3_tts_speech_tokenizer_config.json"


def test_speech_tokenizer_constructs_from_its_checkpoint_configuration():
    from abstractvoice.qwen3_tts.configuration_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Config
    from abstractvoice.qwen3_tts.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Model

    config = Qwen3TTSTokenizerV2Config(**json.loads(CONFIG.read_text()))
    with torch.device("meta"):  # shapes only: no memory, no time spent on random init
        model = Qwen3TTSTokenizerV2Model(config)

    # The encoder keeps only its encoding half; the decoder-side modules the base class would build
    # are released at construction.
    assert model.encoder.upsample is None
    assert model.encoder.decoder_transformer is None
    assert model.encoder.decoder is None
    assert sum(p.numel() for p in model.parameters()) > 100_000_000
