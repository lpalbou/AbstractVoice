"""Dependency compatibility checker for AbstractVoice.

The checker is intentionally lightweight and offline-capable. It uses installed
package metadata first, then falls back to importing modules only when metadata
is unavailable.
"""

from __future__ import annotations

import importlib
from importlib import metadata as importlib_metadata
import re
import sys
from typing import Any, Dict, List, Optional, Tuple


class DependencyChecker:
    """Check and validate AbstractVoice dependencies."""

    CORE_DEPS = {
        "numpy": ("1.24.0", None),
        "requests": ("2.31.0", None),
        "appdirs": ("1.4.0", None),
    }

    LOCAL_TTS_DEPS = {
        "piper-tts": ("1.2.0", None),
        "onnxruntime": ("1.19.0", None),
    }

    LOCAL_STT_DEPS = {
        "faster-whisper": ("0.10.0", None),
        "soundfile": ("0.12.1", None),
    }

    AUDIO_IO_DEPS = {
        "sounddevice": ("0.4.6", None),
        "soundfile": ("0.12.1", None),
        "webrtcvad": ("2.0.10", None),
    }

    # PyTorch is optional in AbstractVoice. Keep these checks broad and focus on
    # known bad combinations rather than stale global upper bounds.
    PYTORCH_COMPAT = {
        "torch": ("2.0.0", None),
        "torchvision": ("0.15.0", None),
        "torchaudio": ("2.0.0", None),
    }

    OPTIONAL_DEPS = {
        "huggingface_hub": ("0.20.0", None),
        "librosa": ("0.10.0", None),
        "f5-tts": ("1.1.0", None),
        "aec-audio-processing": ("1.0.1", None),
        "omnivoice": ("0.1.2", None),
        "transformers": ("4.55.4", None),
        "accelerate": ("1.0.0", None),
        "safetensors": ("0.4.0", None),
        "einops": ("0.8.0", None),
        "sentencepiece": ("0.1.99", None),
        "av": ("14.0.0", None),
        "audioread": ("3.0.0", None),
        "pillow": ("11.0.0", None),
    }

    MODULE_ALIASES = {
        "faster-whisper": "faster_whisper",
        "piper-tts": "piper",
        "f5-tts": "f5_tts",
        "aec-audio-processing": "aec_audio_processing",
        "pillow": "PIL",
    }

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results: Dict[str, Any] = {}

    def _parse_version(self, version_str: str) -> Tuple[int, ...]:
        """Parse a version string into comparable integer components."""
        try:
            raw = str(version_str or "").split("+", 1)[0]
            parts = re.findall(r"\d+", raw)
            return tuple(int(part) for part in parts) if parts else (0,)
        except Exception:
            return (0,)

    def _check_version_range(self, current: str, min_ver: Optional[str], max_ver: Optional[str]) -> bool:
        """Check if current version is within the specified range."""
        if not re.search(r"\d", str(current or "")):
            # Installed but not version-comparable; avoid false alarms.
            return True

        current_tuple = self._parse_version(current)

        if min_ver and current_tuple < self._parse_version(min_ver):
            return False

        if max_ver and current_tuple >= self._parse_version(max_ver):
            return False

        return True

    def _metadata_version(self, package_name: str) -> str | None:
        try:
            return importlib_metadata.version(package_name)
        except importlib_metadata.PackageNotFoundError:
            return None
        except Exception:
            return None

    def _module_version(self, package_name: str) -> str | None:
        module_name = self.MODULE_ALIASES.get(package_name, package_name.replace("-", "_"))
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            return None

        version = getattr(module, "__version__", None)
        if version is None:
            version = getattr(module, "version", None)
        if version is None:
            version = getattr(module, "VERSION", None)
        if callable(version):
            try:
                version = version()
            except Exception:
                version = None
        return str(version or "installed")

    def _check_package(self, package_name: str, min_ver: Optional[str], max_ver: Optional[str]) -> Dict[str, Any]:
        """Check a single package installation and version."""
        try:
            version = self._metadata_version(package_name)
            if version is None:
                version = self._module_version(package_name)

            if version is None:
                return {
                    "status": "missing",
                    "version": None,
                    "compatible": False,
                    "min_version": min_ver,
                    "max_version": max_ver,
                }

            compatible = self._check_version_range(str(version), min_ver, max_ver)
            return {
                "status": "installed",
                "version": str(version),
                "compatible": compatible,
                "min_version": min_ver,
                "max_version": max_ver,
            }

        except Exception as e:
            return {
                "status": "error",
                "version": None,
                "compatible": False,
                "error": str(e),
                "min_version": min_ver,
                "max_version": max_ver,
            }

    def check_core_dependencies(self) -> Dict[str, Any]:
        """Check lightweight base dependencies (always required)."""
        return {
            package: self._check_package(package, min_ver, max_ver)
            for package, (min_ver, max_ver) in self.CORE_DEPS.items()
        }

    def check_local_tts_dependencies(self) -> Dict[str, Any]:
        """Check local TTS dependencies (`abstractvoice[piper]`, `[supertonic]`, `[apple]`, or `[gpu]`)."""
        return {
            package: self._check_package(package, min_ver, max_ver)
            for package, (min_ver, max_ver) in self.LOCAL_TTS_DEPS.items()
        }

    def check_local_stt_dependencies(self) -> Dict[str, Any]:
        """Check local STT dependencies (`abstractvoice[stt]`, `[apple]`, or `[gpu]`)."""
        return {
            package: self._check_package(package, min_ver, max_ver)
            for package, (min_ver, max_ver) in self.LOCAL_STT_DEPS.items()
        }

    def check_audio_io_dependencies(self) -> Dict[str, Any]:
        """Check microphone/playback/VAD dependencies (`abstractvoice[audio-io]`, `[apple]`, or `[gpu]`)."""
        return {
            package: self._check_package(package, min_ver, max_ver)
            for package, (min_ver, max_ver) in self.AUDIO_IO_DEPS.items()
        }

    def check_pytorch_ecosystem(self) -> Dict[str, Any]:
        """Report optional PyTorch ecosystem packages."""
        return {
            package: self._check_package(package, min_ver, max_ver)
            for package, (min_ver, max_ver) in self.PYTORCH_COMPAT.items()
        }

    def check_optional_dependencies(self) -> Dict[str, Any]:
        """Check optional feature dependencies."""
        return {
            package: self._check_package(package, min_ver, max_ver)
            for package, (min_ver, max_ver) in self.OPTIONAL_DEPS.items()
        }

    def check_pytorch_conflicts(self) -> List[str]:
        """Detect specific PyTorch/TorchVision conflicts."""
        conflicts: List[str] = []
        torch_info = self._check_package("torch", "2.0.0", None)
        tv_info = self._check_package("torchvision", "0.15.0", None)

        if torch_info["status"] == "installed" and tv_info["status"] == "installed":
            torch_version = self._parse_version(str(torch_info.get("version") or ""))
            tv_version = self._parse_version(str(tv_info.get("version") or ""))
            minimum_tv_by_torch = [
                ((2, 8), (0, 23)),
                ((2, 7), (0, 22)),
                ((2, 6), (0, 21)),
                ((2, 5), (0, 20)),
                ((2, 4), (0, 19)),
                ((2, 3), (0, 18)),
            ]
            for torch_min, tv_min in minimum_tv_by_torch:
                if torch_version >= torch_min and tv_version < tv_min:
                    conflicts.append(
                        f"PyTorch {torch_info['version']} is likely incompatible with "
                        f"TorchVision {tv_info['version']} (expected torchvision >= {'.'.join(map(str, tv_min))})."
                    )
                    break

            try:
                import torch
                from torchvision.ops import nms

                boxes = torch.tensor([[0, 0, 1, 1]], dtype=torch.float32)
                scores = torch.tensor([1.0], dtype=torch.float32)
                nms(boxes, scores, 0.5)
            except Exception as e:
                msg = str(e)
                if "torchvision::nms" in msg or "operator torchvision" in msg:
                    conflicts.append("TorchVision NMS operator is unavailable; torch/torchvision versions may be mismatched.")

        return conflicts

    def _device_channel_count(self, device: Any, key: str) -> int:
        try:
            getter = getattr(device, "get", None)
            if callable(getter):
                return int(getter(key, 0) or 0)
            return int(getattr(device, key, 0) or 0)
        except Exception:
            return 0

    def check_audio_devices(self) -> Dict[str, Any]:
        """Check whether sounddevice can see usable audio devices."""
        sd_info = self._check_package("sounddevice", "0.4.6", None)
        if sd_info["status"] != "installed":
            return {"status": "skipped", "reason": "sounddevice is not installed"}

        try:
            import sounddevice as sd

            devices = list(sd.query_devices() or [])
            has_input = any(self._device_channel_count(device, "max_input_channels") > 0 for device in devices)
            has_output = any(self._device_channel_count(device, "max_output_channels") > 0 for device in devices)
            status = "ok" if has_input and has_output else "warning"
            return {
                "status": status,
                "device_count": len(devices),
                "has_input": bool(has_input),
                "has_output": bool(has_output),
                "default_device": getattr(getattr(sd, "default", None), "device", None),
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def check_all(self) -> Dict[str, Any]:
        """Run comprehensive dependency check."""
        results = {
            "core": self.check_core_dependencies(),
            "local_tts": self.check_local_tts_dependencies(),
            "local_stt": self.check_local_stt_dependencies(),
            "audio_io": self.check_audio_io_dependencies(),
            "pytorch": self.check_pytorch_ecosystem(),
            "optional": self.check_optional_dependencies(),
            "conflicts": self.check_pytorch_conflicts(),
            "audio": self.check_audio_devices(),
            "python_version": sys.version,
            "platform": sys.platform,
        }

        self.results = results
        return results

    def _status_label(self, info: Dict[str, Any], *, optional: bool = False) -> str:
        status = info.get("status")
        if status == "installed" and info.get("compatible"):
            return "OK"
        if status == "installed":
            return "WARN"
        if status == "missing" and optional:
            return "optional"
        if status == "missing":
            return "MISSING"
        return "ERROR"

    def print_report(self, results: Optional[Dict[str, Any]] = None):
        """Print a formatted dependency report."""
        if results is None:
            results = self.results

        print("AbstractVoice Dependency Check Report")
        print("=" * 50)
        print(f"\nPython: {results['python_version']}")
        print(f"Platform: {results['platform']}")

        print("\nLightweight base dependencies:")
        for package, info in results["core"].items():
            version_info = f"v{info['version']}" if info.get("version") else "not installed"
            print(f"  [{self._status_label(info)}] {package}: {version_info}")

        print("\nLocal TTS dependencies (optional: abstractvoice[piper], [supertonic], [apple], or [gpu]):")
        for package, info in results.get("local_tts", {}).items():
            version_info = f"v{info['version']}" if info.get("version") else "not installed"
            print(f"  [{self._status_label(info, optional=True)}] {package}: {version_info}")

        print("\nLocal STT dependencies (optional: abstractvoice[stt], [apple], or [gpu]):")
        for package, info in results.get("local_stt", {}).items():
            version_info = f"v{info['version']}" if info.get("version") else "not installed"
            print(f"  [{self._status_label(info, optional=True)}] {package}: {version_info}")

        print("\nAudio I/O dependencies (optional: abstractvoice[audio-io], [apple], or [gpu]):")
        for package, info in results.get("audio_io", {}).items():
            version_info = f"v{info['version']}" if info.get("version") else "not installed"
            print(f"  [{self._status_label(info, optional=True)}] {package}: {version_info}")

        print("\nOptional PyTorch ecosystem:")
        for package, info in results["pytorch"].items():
            version_info = f"v{info['version']}" if info.get("version") else "not installed"
            print(f"  [{self._status_label(info, optional=True)}] {package}: {version_info}")

        if results["conflicts"]:
            print("\nDetected PyTorch conflicts:")
            for conflict in results["conflicts"]:
                print(f"  [WARN] {conflict}")

        print("\nOptional feature dependencies:")
        for package, info in results["optional"].items():
            version_info = f"v{info['version']}" if info.get("version") else "not installed"
            print(f"  [{self._status_label(info, optional=True)}] {package}: {version_info}")

        audio = results.get("audio") or {}
        print("\nAudio devices:")
        if audio.get("status") == "ok":
            print(f"  [OK] input/output devices detected ({audio.get('device_count', 0)} total)")
        elif audio.get("status") == "warning":
            print(
                "  [WARN] sounddevice is installed, but usable input/output devices were not both detected "
                f"({audio.get('device_count', 0)} total)"
            )
        elif audio.get("status") == "skipped":
            print(f"  [optional] {audio.get('reason')}")
        else:
            print(f"  [WARN] audio device query failed: {audio.get('error', 'unknown error')}")

        recommendations: List[str] = []
        if any(info.get("status") != "installed" or not info.get("compatible") for info in results["core"].values()):
            recommendations.append("Reinstall or upgrade the base package: pip install --upgrade abstractvoice")
        local_voice_missing = any(
            info.get("status") != "installed"
            for group in ("local_tts", "local_stt", "audio_io")
            for info in results.get(group, {}).values()
        )
        if local_voice_missing:
            recommendations.append(
                "For a platform local stack, install: pip install \"abstractvoice[apple]\" or \"abstractvoice[gpu]\". "
                "For smaller installs, compose granular extras such as \"abstractvoice[supertonic,stt,audio-io]\"."
            )
        if results["conflicts"]:
            recommendations.append("Align torch, torchaudio, and torchvision versions for the optional engine you are using.")
        if audio.get("status") in ("warning", "error"):
            recommendations.append("Check microphone/speaker permissions, default audio devices, and PortAudio/sounddevice setup.")

        print("\nRecommendations:")
        if recommendations:
            for rec in recommendations:
                print(f"  - {rec}")
        else:
            print("  - No blocking dependency issues detected.")

        print("\n" + "=" * 50)


def check_dependencies(verbose: bool = True) -> Dict[str, Any]:
    """Quick function to check all dependencies."""
    checker = DependencyChecker(verbose=verbose)
    results = checker.check_all()
    if verbose:
        checker.print_report(results)
    return results


if __name__ == "__main__":
    check_dependencies()
