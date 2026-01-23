"""Public `VoiceManager` façade.

Implementation is split into small focused modules under `abstractvoice/vm/`
to keep files readable and responsibilities clear.
"""

from .vm.manager import VoiceManager

__all__ = ["VoiceManager"]

