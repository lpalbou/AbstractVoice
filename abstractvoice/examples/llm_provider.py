"""
LLM provider abstraction for OpenAI-compatible local APIs (Ollama, LMStudio, etc.).

Both Ollama and LMStudio expose the same OpenAI-compatible surface:
  - POST /v1/chat/completions  (chat)
  - GET  /v1/models            (model listing)

A provider is a (name, base_url) pair — no SDK dependency required.
"""

from __future__ import annotations

import re
from typing import Any

import requests


_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think\s*>\s*", flags=re.IGNORECASE | re.DOTALL)
_THINK_OPEN_RE = re.compile(r"<think\b[^>]*>", flags=re.IGNORECASE | re.DOTALL)
_THINK_TAG_RE = re.compile(r"</?think\b[^>]*>", flags=re.IGNORECASE | re.DOTALL)
_RE_MANY_BLANK_LINES = re.compile(r"\n{3,}")


def strip_think_blocks(text: str) -> str:
    """Remove local-LLM `<think>...</think>` blocks before display or speech."""
    s = str(text or "")
    if not s:
        return ""
    if "<think" not in s.lower():
        return s.strip()

    out = _THINK_BLOCK_RE.sub("", s)
    m = _THINK_OPEN_RE.search(out)
    if m is not None and "</think" not in out[m.end() :].lower():
        out = out[: m.start()]
    out = _THINK_TAG_RE.sub("", out)
    out = _RE_MANY_BLANK_LINES.sub("\n\n", out)
    return out.strip()


def _extract_chat_text(data: Any, *, fallback: str = "") -> str:
    """Best-effort extraction from OpenAI-compatible and Ollama-native shapes."""
    if not isinstance(data, dict):
        return str(fallback or "").strip()

    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0] if isinstance(choices[0], dict) else {}
        message = first.get("message") if isinstance(first, dict) else None
        if isinstance(message, dict) and "content" in message:
            return str(message.get("content") or "").strip()
        text = first.get("text") if isinstance(first, dict) else None
        if text is not None:
            return str(text or "").strip()

    message = data.get("message")
    if isinstance(message, dict) and "content" in message:
        return str(message.get("content") or "").strip()

    return str(fallback or data).strip()


class LLMProvider:
    """Configuration for an OpenAI-compatible LLM API endpoint."""

    def __init__(self, name: str, base_url: str) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")

    # -- endpoints -----------------------------------------------------------

    @property
    def chat_url(self) -> str:
        return f"{self.base_url}/v1/chat/completions"

    @property
    def models_url(self) -> str:
        return f"{self.base_url}/v1/models"

    # -- helpers -------------------------------------------------------------

    def list_models(self, timeout: float = 5.0) -> list[str]:
        """Fetch available model ids from the provider (empty list on failure)."""
        try:
            resp = requests.get(self.models_url, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            models = data.get("data", []) if isinstance(data, dict) else []
            return sorted(str(m["id"]) for m in models if isinstance(m, dict) and m.get("id"))
        except Exception:
            return []

    def is_reachable(self, timeout: float = 3.0) -> bool:
        try:
            resp = requests.get(self.models_url, timeout=timeout)
            return resp.status_code == 200
        except Exception:
            return False

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.4,
        max_tokens: int = 1024,
        timeout: float | tuple[float, float] = (5.0, 600.0),
    ) -> dict[str, Any]:
        """Run one non-streaming OpenAI-compatible chat completion.

        This intentionally avoids provider SDKs so examples can point at Ollama,
        LM Studio, or another compatible local proxy with the same tiny client.
        """
        payload = {
            "model": str(model or "").strip(),
            "messages": list(messages or []),
            "stream": False,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }
        resp = requests.post(self.chat_url, json=payload, timeout=timeout)
        resp.raise_for_status()

        try:
            data = resp.json()
        except Exception:
            data = {}

        usage = data.get("usage") if isinstance(data, dict) and isinstance(data.get("usage"), dict) else {}
        text = strip_think_blocks(_extract_chat_text(data, fallback=resp.text))
        return {"text": text, "usage": dict(usage), "raw": data if isinstance(data, dict) else None}

    def __repr__(self) -> str:
        return f"LLMProvider({self.name!r}, {self.base_url!r})"


# -- presets -----------------------------------------------------------------

PROVIDER_PRESETS: dict[str, LLMProvider] = {
    "ollama": LLMProvider("ollama", "http://localhost:11434"),
    "lmstudio": LLMProvider("lmstudio", "http://localhost:1234"),
}

DEFAULT_PROVIDER = "ollama"
DEFAULT_MODEL = "gemma3:1b"


def resolve_provider(name_or_url: str | None) -> LLMProvider:
    """Resolve a preset name or treat the string as a custom base URL."""
    raw = str(name_or_url or DEFAULT_PROVIDER).strip() or DEFAULT_PROVIDER
    key = raw.lower()
    if key in PROVIDER_PRESETS:
        return PROVIDER_PRESETS[key]
    return LLMProvider("custom", raw)
