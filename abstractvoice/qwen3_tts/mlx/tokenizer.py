"""Qwen2 text tokenizer for the MLX path, built from the snapshot's vocab.json + merges.txt.

The Qwen3-TTS snapshots ship a slow-tokenizer layout (``vocab.json``, ``merges.txt``,
``tokenizer_config.json`` with ``added_tokens_decoder``; no ``tokenizer.json``). This module
assembles the same fast tokenizer Transformers' ``Qwen2Converter`` builds, with the ``tokenizers``
library alone: byte-level BPE, NFC normalisation, the Qwen2 pre-tokenisation regex, and the
snapshot's added special tokens. ``tests/test_qwen3_tts_mlx_tokenizer.py`` checks token identity
against the Transformers tokenizer on the prompt formats the model consumes.
"""

from __future__ import annotations

import json
import os
from typing import List

# Qwen2's pre-tokenisation pattern (transformers.convert_slow_tokenizer.Qwen2Converter).
QWEN2_PRETOKENIZER_PATTERN = (
    r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
)


class Qwen3TTSTextTokenizer:
    """``encode(text) -> list[int]`` over the snapshot's Qwen2 vocabulary (no special-token insertion)."""

    def __init__(self, snapshot_dir: str) -> None:
        try:
            from tokenizers import AddedToken, Regex, Tokenizer, decoders, normalizers, pre_tokenizers, processors
            from tokenizers.models import BPE
        except Exception as e:  # pragma: no cover - import guard
            raise RuntimeError(
                "The Qwen3-TTS MLX backend needs the `tokenizers` package for its text tokenizer.\n"
                'Install with:\n  pip install "abstractvoice[qwen3-tts]"'
            ) from e

        vocab_path = os.path.join(snapshot_dir, "vocab.json")
        merges_path = os.path.join(snapshot_dir, "merges.txt")
        config_path = os.path.join(snapshot_dir, "tokenizer_config.json")
        for path in (vocab_path, merges_path, config_path):
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Qwen3-TTS snapshot is missing {os.path.basename(path)}: {path}")

        with open(vocab_path, "r", encoding="utf-8") as fh:
            vocab = json.load(fh)
        merges: List[tuple] = []
        with open(merges_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if not line or line.startswith("#version"):
                    continue
                parts = line.split(" ")
                if len(parts) != 2:
                    raise ValueError(f"malformed merge rule in {merges_path}: {line!r}")
                merges.append((parts[0], parts[1]))
        with open(config_path, "r", encoding="utf-8") as fh:
            tok_config = json.load(fh)

        tokenizer = Tokenizer(
            BPE(
                vocab=vocab,
                merges=merges,
                dropout=None,
                unk_token=None,
                continuing_subword_prefix="",
                end_of_word_suffix="",
                fuse_unk=False,
                byte_fallback=False,
            )
        )
        tokenizer.normalizer = normalizers.NFC()
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(Regex(QWEN2_PRETOKENIZER_PATTERN), behavior="isolated", invert=False),
                pre_tokenizers.ByteLevel(add_prefix_space=bool(tok_config.get("add_prefix_space", False)), use_regex=False),
            ]
        )
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

        added = []
        for token_id, spec in sorted(
            (int(k), v) for k, v in (tok_config.get("added_tokens_decoder") or {}).items()
        ):
            content = str(spec["content"])
            # Added tokens extend the vocabulary; their ids must match the checkpoint's.
            existing = vocab.get(content)
            if existing is not None and int(existing) != token_id:
                raise ValueError(f"added token {content!r} id {token_id} conflicts with vocab id {existing}")
            added.append(
                AddedToken(
                    content,
                    single_word=bool(spec.get("single_word", False)),
                    lstrip=bool(spec.get("lstrip", False)),
                    rstrip=bool(spec.get("rstrip", False)),
                    normalized=bool(spec.get("normalized", False)),
                    special=bool(spec.get("special", True)),
                )
            )
        if added:
            tokenizer.add_tokens(added)
            for token_id, spec in sorted((int(k), v) for k, v in tok_config["added_tokens_decoder"].items()):
                got = tokenizer.token_to_id(str(spec["content"]))
                if got != token_id:
                    raise ValueError(
                        f"added token {spec['content']!r} resolved to id {got}, checkpoint expects {token_id}; "
                        "the vocabulary layout differs from the one this tokenizer builder understands"
                    )
        self._tokenizer = tokenizer

    def encode(self, text: str) -> List[int]:
        return list(self._tokenizer.encode(str(text), add_special_tokens=False).ids)

    def decode(self, ids) -> str:
        return self._tokenizer.decode([int(i) for i in ids], skip_special_tokens=False)
