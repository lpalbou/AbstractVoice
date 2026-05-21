from __future__ import annotations

import subprocess
import sys
import textwrap


def _run_blocked_import_smoke(code: str) -> subprocess.CompletedProcess[str]:
    blocked = (
        "piper",
        "faster_whisper",
        "sounddevice",
        "soundfile",
        "webrtcvad",
        "onnxruntime",
        "omnivoice",
        "torch",
        "torchaudio",
        "transformers",
    )
    harness = f"""
import importlib.abc
import sys

BLOCKED = {blocked!r}

class _Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        root = str(fullname).split(".", 1)[0]
        if root in BLOCKED:
            raise ImportError(f"blocked optional local dependency: {{fullname}}")
        return None

sys.meta_path.insert(0, _Blocker())

{textwrap.dedent(code)}
"""
    # macOS + native deps (torch, audio backends, etc.) can make `vfork()`-based
    # subprocess launching unstable inside long-running pytest sessions. Use
    # the documented CPython escape hatch to disable `vfork()` for these smoke
    # subprocesses so the import-boundary tests don't crash the suite.
    old_vfork = getattr(subprocess, "_USE_VFORK", None)
    old_spawn = getattr(subprocess, "_USE_POSIX_SPAWN", None)
    try:
        if hasattr(subprocess, "_USE_VFORK"):
            subprocess._USE_VFORK = False  # type: ignore[attr-defined]
        if hasattr(subprocess, "_USE_POSIX_SPAWN"):
            subprocess._USE_POSIX_SPAWN = False  # type: ignore[attr-defined]
        return subprocess.run(
            [sys.executable, "-c", harness],
            check=False,
            text=True,
            capture_output=True,
        )
    finally:
        if old_vfork is not None and hasattr(subprocess, "_USE_VFORK"):
            subprocess._USE_VFORK = old_vfork  # type: ignore[attr-defined]
        if old_spawn is not None and hasattr(subprocess, "_USE_POSIX_SPAWN"):
            subprocess._USE_POSIX_SPAWN = old_spawn  # type: ignore[attr-defined]


def test_import_abstractvoice_and_plugin_without_local_voice_deps() -> None:
    result = _run_blocked_import_smoke(
        """
        import abstractvoice
        import abstractvoice.integrations.abstractcore_plugin
        import importlib.metadata as metadata

        all_eps = metadata.entry_points()
        if hasattr(all_eps, "select"):
            eps = all_eps.select(group="abstractcore.capabilities_plugins")
        else:
            eps = all_eps.get("abstractcore.capabilities_plugins", [])
        assert any(ep.name == "abstractvoice" for ep in eps)
        print("ok")
        """
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert result.stdout.strip() == "ok"


def test_remote_voicemanager_constructs_without_local_voice_deps() -> None:
    result = _run_blocked_import_smoke(
        """
        from abstractvoice import VoiceManager

        vm = VoiceManager(
            tts_engine="openai-compatible",
            stt_engine="openai-compatible",
            remote_base_url="http://remote.test/v1",
            remote_api_key="sk-test",
            allow_downloads=False,
        )
        assert vm.tts_adapter is not None
        assert getattr(vm.tts_adapter, "engine_id", "") == "openai-compatible"
        assert vm.cloning_engine == "omnivoice"
        print("ok")
        """
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert result.stdout.strip() == "ok"


def test_default_voicemanager_constructs_without_local_voice_deps() -> None:
    result = _run_blocked_import_smoke(
        """
        from abstractvoice import VoiceManager

        vm = VoiceManager(
            remote_api_key="sk-test",
            allow_downloads=False,
        )
        assert vm.tts_adapter is not None
        assert getattr(vm.tts_adapter, "engine_id", "") == "openai"
        assert vm.cloning_engine == "omnivoice"
        print("ok")
        """
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert result.stdout.strip() == "ok"
