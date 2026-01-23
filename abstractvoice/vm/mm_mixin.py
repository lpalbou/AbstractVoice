"""Model-management façade methods for VoiceManager.

Terminology:
- MM = Model Management
"""

from __future__ import annotations


class MMMixin:
    """Thin wrappers to keep `VoiceManager` API stable and simple."""

    def list_available_models(self, language: str = None) -> dict:
        from ..simple_model_manager import get_model_manager

        manager = get_model_manager(self.debug_mode)
        return manager.list_available_models(language)

    def download_model(self, model_name: str, progress_callback=None) -> bool:
        from ..simple_model_manager import download_model

        return download_model(model_name, progress_callback)

    def is_model_ready(self) -> bool:
        from ..simple_model_manager import is_ready

        return is_ready()

    def ensure_ready(self, auto_download: bool = True) -> bool:
        if self.is_model_ready():
            return True
        if not auto_download:
            return False

        from ..simple_model_manager import get_model_manager

        manager = get_model_manager(self.debug_mode)
        return manager.download_essential_model()

    def get_cache_status(self) -> dict:
        from ..simple_model_manager import get_model_manager

        manager = get_model_manager(self.debug_mode)
        return manager.get_status()

