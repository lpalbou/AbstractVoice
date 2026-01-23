"""Optional voice cloning support (behind `abstractvoice[cloning]`)."""

from .store import VoiceCloneStore
from .manager import VoiceCloner

__all__ = ["VoiceCloneStore", "VoiceCloner"]

