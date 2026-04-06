"""
LLM provider abstraction for OpenAI-compatible local APIs (Ollama, LMStudio, etc.).

Both Ollama and LMStudio expose the same OpenAI-compatible surface:
  - POST /v1/chat/completions  (chat)
  - GET  /v1/models            (model listing)

A provider is a (name, base_url) pair — no SDK dependency required.
"""

from __future__ import annotations

import requests


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
            return sorted(m["id"] for m in data.get("data", []))
        except Exception:
            return []

    def is_reachable(self, timeout: float = 3.0) -> bool:
        try:
            resp = requests.get(self.models_url, timeout=timeout)
            return resp.status_code == 200
        except Exception:
            return False

    def __repr__(self) -> str:
        return f"LLMProvider({self.name!r}, {self.base_url!r})"


# -- presets -----------------------------------------------------------------

PROVIDER_PRESETS: dict[str, LLMProvider] = {
    "ollama": LLMProvider("ollama", "http://localhost:11434"),
    "lmstudio": LLMProvider("lmstudio", "http://localhost:1234"),
}

DEFAULT_PROVIDER = "ollama"
DEFAULT_MODEL = "gemma3:1b"


def resolve_provider(name_or_url: str) -> LLMProvider:
    """Resolve a preset name or treat the string as a custom base URL."""
    key = name_or_url.strip().lower()
    if key in PROVIDER_PRESETS:
        return PROVIDER_PRESETS[key]
    return LLMProvider("custom", name_or_url)
