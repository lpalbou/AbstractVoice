"""The MLX path's Qwen2 tokenizer (built with `tokenizers` from vocab.json + merges.txt) encodes
exactly like the Transformers tokenizer the PyTorch path uses.

Needs a cached Qwen3-TTS snapshot and the `tokenizers` package; the identity check also needs
`transformers`. Skips cleanly otherwise (no download, no model load, no mlx import).
"""

from __future__ import annotations

import importlib.util

import pytest

from abstractvoice.local_models import hf_cached_snapshot_dir

MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"

PROMPTS = [
    "<|im_start|>assistant\nWhat to expect. The first speak takes about thirty seconds because the weights load onto the Apple GPU.<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nA warm, clear narrator.<|im_end|>\n",
    "<|im_start|>assistant\n我叫通义千问，是阿里云的开源大模型。<|im_end|>\n",
    "Numbers 12345 and 3.14159, émojis 🤖, ß, Ça va? It's 5 o'clock — don't you think?\n\nNew paragraph\twith tabs.  Double  spaces.",
    "日本語のテキストと한국어 텍스트, Русский текст, Deutsch mit Umlauten äöü.",
    "   leading spaces and trailing spaces   ",
    "<tts_pad><tts_text_bos><tts_text_bos_single><|audio_pad|><|endoftext|>",
    "A line\nwith\r\nmixed newlines\n",
]


@pytest.fixture(scope="module")
def snapshot():
    if importlib.util.find_spec("tokenizers") is None:
        pytest.skip("tokenizers is not installed")
    snap = hf_cached_snapshot_dir(MODEL)
    if snap is None:
        pytest.skip(f"{MODEL} is not in the local Hugging Face cache")
    return str(snap)


def test_special_tokens_resolve_to_the_checkpoint_ids(snapshot):
    from abstractvoice.qwen3_tts.mlx.tokenizer import Qwen3TTSTextTokenizer

    tok = Qwen3TTSTextTokenizer(snapshot)
    assert tok.encode("<|im_start|>") == [151644]
    assert tok.encode("<|im_end|>") == [151645]
    assert tok.encode("<tts_pad>") == [151671]
    assert tok.encode("<|im_start|>assistant\n")[:3] == [151644, 77091, 198]
    assert tok.decode(tok.encode("Hello world.")) == "Hello world."


def test_encodes_token_identical_to_transformers(snapshot):
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    from abstractvoice.qwen3_tts.mlx.tokenizer import Qwen3TTSTextTokenizer

    ours = Qwen3TTSTextTokenizer(snapshot)
    reference = AutoTokenizer.from_pretrained(snapshot)
    for text in PROMPTS:
        assert ours.encode(text) == reference(text, add_special_tokens=False)["input_ids"], text
