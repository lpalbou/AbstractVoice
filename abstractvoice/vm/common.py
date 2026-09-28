"""Common helpers for VoiceManager parts.

This module exists to avoid circular imports while keeping `voice_manager.py`
small and focused on the public façade.
"""

from __future__ import annotations

def remote_endpoint_kwargs(vm, provider) -> dict:
    """``{"base_url", "api_key"}`` for a remote adapter of ``provider`` built by ``vm``.

    See ``adapters.openai_compatible_http.remote_endpoint``; ``auto`` resolves to
    ``openai`` like the TTS registry does. Local engines ignore both values.
    """
    from ..adapters.openai_compatible_http import remote_endpoint

    name = str(provider or "").strip().lower().replace("_", "-")
    base_url, api_key = remote_endpoint(
        "openai" if name in ("", "auto") else name,
        remote_base_url=getattr(vm, "remote_base_url", None),
        remote_api_key=getattr(vm, "remote_api_key", None),
        openai_base_url=getattr(vm, "openai_base_url", None),
        openai_api_key=getattr(vm, "openai_api_key", None),
    )
    return {"base_url": base_url, "api_key": api_key}


def import_voice_recognizer():
    """Import VoiceRecognizer with a helpful error if dependencies are missing."""
    try:
        from ..recognition import VoiceRecognizer
        return VoiceRecognizer
    except ImportError as e:
        raise ImportError(
            "Microphone capture/listen() requires optional dependencies to be installed correctly.\n"
            "Try:\n"
            "  pip install --upgrade abstractvoice\n"
            f"Original error: {e}"
        ) from e

