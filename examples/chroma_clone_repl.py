"""Minimal REPL-style smoke test for Chroma cloning.

Run:
  python -i examples/chroma_clone_repl.py

Prereqs:
  - pip install -e '.[chroma]'
  - abstractvoice-prefetch --chroma
"""

from __future__ import annotations

import atexit
import time
from pathlib import Path

from abstractvoice import VoiceManager


ref = Path("audio_samples/hal9000/hal9000_hello.wav")
if not ref.exists():
    raise SystemExit(f"Missing reference audio: {ref}")

vm = VoiceManager(cloning_engine="chroma", debug_mode=True, allow_downloads=False)
atexit.register(vm.cleanup)

vid = vm.clone_voice(str(ref), reference_text="Hello.", engine="chroma")
print("voice_id:", vid)

vm.speak("Good evening, Dave.", voice=vid)
# Wait for synthesis to start (speak() is async for cloned voices).
try:
    for _ in range(200):  # ~20s max
        if getattr(vm, "_cloned_synthesis_active", None) and vm._cloned_synthesis_active.is_set():
            break
        time.sleep(0.1)
except Exception:
    pass

# Wait for synthesis to finish.
try:
    while getattr(vm, "_cloned_synthesis_active", None) and vm._cloned_synthesis_active.is_set():
        time.sleep(0.1)
except Exception:
    pass

# Wait for playback to drain (best-effort).
while vm.is_speaking():
    time.sleep(0.1)

print("cloning_runtime_info:", vm.get_cloning_runtime_info())

print("\nReady. Try in the REPL:")
print("  vm.speak('Test phrase', voice=vid)")
print("  vm.speak_to_file('Test phrase', 'out.wav', voice=vid)")
