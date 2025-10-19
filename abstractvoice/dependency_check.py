"""Dependency compatibility checker for AbstractVoice.

This module provides utilities to check dependency versions and compatibility,
helping users diagnose and resolve installation issues.
"""

import sys
import importlib
from typing import Dict, List, Tuple, Optional
import warnings


class DependencyChecker:
    """Check and validate AbstractVoice dependencies."""

    # Known compatible version ranges
    PYTORCH_COMPAT = {
        "torch": ("2.0.0", "2.4.0"),
        "torchvision": ("0.15.0", "0.19.0"),
        "torchaudio": ("2.0.0", "2.4.0"),
    }

    CORE_DEPS = {
        "numpy": ("1.24.0", None),
        "requests": ("2.31.0", None),
    }

    OPTIONAL_DEPS = {
        "coqui-tts": ("0.27.0", "0.30.0"),
        "openai-whisper": ("20230314", None),
        "sounddevice": ("0.4.6", None),
        "librosa": ("0.10.0", "0.11.0"),
        "flask": ("2.0.0", None),
        "webrtcvad": ("2.0.10", None),
        "PyAudio": ("0.2.13", None),
        "soundfile": ("0.12.1", None),
        "tiktoken": ("0.6.0", None),
    }

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results = {}

    def _parse_version(self, version_str: str) -> Tuple[int, ...]:
        """Parse version string into tuple of integers for comparison."""
        try:
            # Handle version strings like "20230314" or "2.0.0"
            if version_str.isdigit():
                return (int(version_str),)

            # Standard semantic versioning
            version_parts = version_str.split('.')
            return tuple(int(part) for part in version_parts if part.isdigit())
        except (ValueError, AttributeError):
            return (0,)

    def _check_version_range(self, current: str, min_ver: Optional[str], max_ver: Optional[str]) -> bool:
        """Check if current version is within the specified range."""
        current_tuple = self._parse_version(current)

        if min_ver:
            min_tuple = self._parse_version(min_ver)
            if current_tuple < min_tuple:
                return False

        if max_ver:
            max_tuple = self._parse_version(max_ver)
            if current_tuple >= max_tuple:
                return False

        return True

    def _check_package(self, package_name: str, min_ver: Optional[str], max_ver: Optional[str]) -> Dict:
        """Check a single package installation and version."""
        try:
            module = importlib.import_module(package_name.replace('-', '_'))
            version = getattr(module, '__version__', 'unknown')

            # Special handling for packages with different version attributes
            if version == 'unknown':
                if hasattr(module, 'version'):
                    version = module.version
                elif hasattr(module, 'VERSION'):
                    version = module.VERSION
                elif package_name == 'openai-whisper':
                    # Whisper uses different versioning
                    try:
                        import whisper
                        version = getattr(whisper, '__version__', 'installed')
                    except:
                        version = 'installed'

            compatible = self._check_version_range(str(version), min_ver, max_ver)

            return {
                'status': 'installed',
                'version': str(version),
                'compatible': compatible,
                'min_version': min_ver,
                'max_version': max_ver
            }

        except ImportError:
            return {
                'status': 'missing',
                'version': None,
                'compatible': False,
                'min_version': min_ver,
                'max_version': max_ver
            }
        except Exception as e:
            return {
                'status': 'error',
                'version': None,
                'compatible': False,
                'error': str(e),
                'min_version': min_ver,
                'max_version': max_ver
            }

    def check_core_dependencies(self) -> Dict:
        """Check core dependencies (always required)."""
        results = {}
        for package, (min_ver, max_ver) in self.CORE_DEPS.items():
            results[package] = self._check_package(package, min_ver, max_ver)
        return results

    def check_pytorch_ecosystem(self) -> Dict:
        """Check PyTorch ecosystem for compatibility issues."""
        results = {}
        for package, (min_ver, max_ver) in self.PYTORCH_COMPAT.items():
            results[package] = self._check_package(package, min_ver, max_ver)
        return results

    def check_optional_dependencies(self) -> Dict:
        """Check optional dependencies."""
        results = {}
        for package, (min_ver, max_ver) in self.OPTIONAL_DEPS.items():
            results[package] = self._check_package(package, min_ver, max_ver)
        return results

    def check_pytorch_conflicts(self) -> List[str]:
        """Detect specific PyTorch/TorchVision conflicts."""
        conflicts = []

        try:
            import torch
            import torchvision

            torch_version = self._parse_version(torch.__version__)
            tv_version = self._parse_version(torchvision.__version__)

            # Check known incompatible combinations
            if torch_version >= (2, 3, 0) and tv_version < (0, 18, 0):
                conflicts.append(f"PyTorch {torch.__version__} is incompatible with TorchVision {torchvision.__version__}")

            # Test torchvision::nms operator availability
            try:
                from torchvision.ops import nms
                # Try to use nms to check if it actually works
                import torch
                boxes = torch.tensor([[0, 0, 1, 1]], dtype=torch.float32)
                scores = torch.tensor([1.0], dtype=torch.float32)
                nms(boxes, scores, 0.5)
            except Exception as e:
                if "torchvision::nms does not exist" in str(e):
                    conflicts.append("TorchVision NMS operator not available - version mismatch detected")

        except ImportError:
            pass  # PyTorch not installed

        return conflicts

    def check_all(self) -> Dict:
        """Run comprehensive dependency check."""
        results = {
            'core': self.check_core_dependencies(),
            'pytorch': self.check_pytorch_ecosystem(),
            'optional': self.check_optional_dependencies(),
            'conflicts': self.check_pytorch_conflicts(),
            'python_version': sys.version,
            'platform': sys.platform
        }

        self.results = results
        return results

    def print_report(self, results: Optional[Dict] = None):
        """Print a formatted dependency report."""
        if results is None:
            results = self.results

        print("🔍 AbstractVoice Dependency Check Report")
        print("=" * 50)

        # Python version
        print(f"\n🐍 Python: {results['python_version']}")
        print(f"🖥️  Platform: {results['platform']}")

        # Core dependencies
        print(f"\n📦 Core Dependencies:")
        for package, info in results['core'].items():
            status_icon = "✅" if info['status'] == 'installed' and info['compatible'] else "❌"
            version_info = f"v{info['version']}" if info['version'] else "not installed"
            print(f"  {status_icon} {package}: {version_info}")

        # PyTorch ecosystem
        print(f"\n🔥 PyTorch Ecosystem:")
        pytorch_all_good = True
        for package, info in results['pytorch'].items():
            status_icon = "✅" if info['status'] == 'installed' and info['compatible'] else "❌"
            version_info = f"v{info['version']}" if info['version'] else "not installed"
            if info['status'] != 'installed' or not info['compatible']:
                pytorch_all_good = False
            print(f"  {status_icon} {package}: {version_info}")

        # Conflicts
        if results['conflicts']:
            print(f"\n⚠️  Detected Conflicts:")
            for conflict in results['conflicts']:
                print(f"  ❌ {conflict}")
            pytorch_all_good = False

        if pytorch_all_good and all(info['status'] == 'installed' for info in results['pytorch'].values()):
            print("  🎉 PyTorch ecosystem looks compatible!")

        # Optional dependencies
        print(f"\n🔧 Optional Dependencies:")
        installed_optional = []
        missing_optional = []

        for package, info in results['optional'].items():
            if info['status'] == 'installed':
                status_icon = "✅" if info['compatible'] else "⚠️"
                installed_optional.append(f"  {status_icon} {package}: v{info['version']}")
            else:
                missing_optional.append(f"  ⭕ {package}: not installed")

        if installed_optional:
            print("  Installed:")
            for line in installed_optional:
                print(line)

        if missing_optional:
            print("  Missing:")
            for line in missing_optional:
                print(line)

        # Recommendations
        print(f"\n💡 Recommendations:")

        if not pytorch_all_good:
            print("  🔧 Fix PyTorch conflicts with:")
            print("     pip uninstall torch torchvision torchaudio")
            print("     pip install abstractvoice[all]")

        if not any(info['status'] == 'installed' for info in results['optional'].values()):
            print("  📦 Install voice functionality with:")
            print("     pip install abstractvoice[voice-full]")

        print("\n" + "=" * 50)


def check_dependencies(verbose: bool = True) -> Dict:
    """Quick function to check all dependencies."""
    checker = DependencyChecker(verbose=verbose)
    results = checker.check_all()
    if verbose:
        checker.print_report(results)
    return results


if __name__ == "__main__":
    check_dependencies()