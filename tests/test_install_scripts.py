"""The standalone one-line installers (scripts/install.sh, scripts/install.ps1).

They are served at https://abstractframework.ai/voice/install.sh and .../install.ps1, so these
checks keep them in step with the package: the default pin is this release, every extra they
install exists in pyproject.toml, both scripts install the same profiles, and the documented
one-liners are the exact URLs.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
SH = ROOT / "scripts" / "install.sh"
PS1 = ROOT / "scripts" / "install.ps1"

SH_LINE = "curl -LsSf https://abstractframework.ai/voice/install.sh | sh"
PS1_LINE = 'powershell -ExecutionPolicy ByPass -c "irm https://abstractframework.ai/voice/install.ps1 | iex"'
PIP_LINES = ('pip install "abstractvoice[apple]"', 'pip install "abstractvoice[gpu]"')


def _sh() -> str:
    return SH.read_text(encoding="utf-8")


def _ps1() -> str:
    return PS1.read_text(encoding="utf-8")


def _extras() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["optional-dependencies"]


def _version() -> str:
    m = re.search(r'__version__\s*=\s*"([^"]+)"', (ROOT / "abstractvoice" / "_version.py").read_text(encoding="utf-8"))
    assert m, "abstractvoice/_version.py has no __version__"
    return m.group(1)


def _one(pattern: str, text: str, what: str) -> str:
    m = re.search(pattern, text, re.MULTILINE)
    assert m, f"{what} not found"
    return m.group(1)


def test_both_scripts_exist():
    assert SH.is_file()
    assert PS1.is_file()


@pytest.mark.skipif(shutil.which("sh") is None, reason="no POSIX sh")
def test_install_sh_parses_and_prints_help():
    subprocess.run(["sh", "-n", str(SH)], check=True)
    out = subprocess.run(["sh", str(SH), "--help"], check=True, capture_output=True, text=True).stdout
    for opt in ("--profile auto|light|apple|gpu", "--pin VERSION|latest", "--from PATH", "--print"):
        assert opt in out


@pytest.mark.skipif(shutil.which("sh") is None, reason="no POSIX sh")
def test_install_sh_cut_mid_download_runs_nothing():
    """The body is one function called on the last line: a truncated `curl | sh` stream executes nothing."""
    text = SH.read_text(encoding="utf-8")
    assert text.rstrip().splitlines()[-1] == 'main "$@"'
    data = text.encode("utf-8")
    for frac in (0.2, 0.5, 0.8, 0.99):
        cut = data[: int(len(data) * frac)]
        out = subprocess.run(["sh", "-s", "--", "--print"], input=cut, capture_output=True).stdout
        assert b"AbstractVoice installer" not in out, f"a stream cut at {frac:.0%} ran the installer"


def test_option_parity_between_scripts():
    sh, ps1 = _sh(), _ps1()
    for opt in ("--profile", "--pin", "--from", "--no-modify-path", "--print", "--dry-run", "--verbose"):
        assert opt in sh, f"install.sh lacks {opt}"
    assert "[Alias('Profile')]" in ps1 and "[Alias('DryRun')]" in ps1
    for param in ("$Pin", "$From", "$NoModifyPath", "$Print"):
        assert f"]{param}" in ps1, f"install.ps1 lacks {param}"
    assert "[CmdletBinding()]" in ps1 and "$VerbosePreference -eq 'Continue'" in ps1  # -Verbose


@pytest.mark.skipif(shutil.which("pwsh") is None, reason="no PowerShell")
def test_install_ps1_parses():
    cmd = (
        "$e = $null; [System.Management.Automation.Language.Parser]::ParseFile("
        f"'{PS1}', [ref]$null, [ref]$e) | Out-Null; $e.Count"
    )
    out = subprocess.run(["pwsh", "-NoProfile", "-Command", cmd], check=True, capture_output=True, text=True).stdout
    assert out.strip() == "0"


def test_default_pin_is_this_release():
    assert _one(r'^AV_PIN_DEFAULT="([^"]+)"', _sh(), "AV_PIN_DEFAULT in install.sh") == _version()
    assert _one(r"^\$AvPinDefault = '([^']+)'", _ps1(), "$AvPinDefault in install.ps1") == _version()


def test_profile_extras_match_between_scripts_and_exist_in_pyproject():
    sh, ps1, extras = _sh(), _ps1(), _extras()
    light_sh = _one(r'^AV_LIGHT_EXTRAS="([^"]+)"', sh, "AV_LIGHT_EXTRAS in install.sh")
    light_ps1 = _one(r"^\$AvLightExtras = '([^']+)'", ps1, "$AvLightExtras in install.ps1")
    arm64_ps1 = _one(r"^\$AvLightExtrasArm64 = '([^']+)'", ps1, "$AvLightExtrasArm64 in install.ps1")
    assert light_sh == light_ps1
    # The profile -> extras mapping each script installs.
    sh_profiles = dict(re.findall(r"(apple|gpu)\) EXTRAS=(\w+)", sh))
    assert sh_profiles == {"apple": "apple", "gpu": "gpu"}
    assert "$extras = if ($profileName -eq 'gpu') { 'gpu' }" in ps1
    for script in (sh, ps1):
        assert "onnx-cuda" in script
        assert "--reinstall-package', 'onnxruntime-gpu'" in script or "--reinstall-package onnxruntime-gpu" in script
    named = set()
    for spec in (light_sh, arm64_ps1, "apple", "gpu", "onnx-cuda"):
        named.update(spec.split(","))
    for text in (sh, ps1):  # every abstractvoice[...] the scripts or their headers name
        for group in re.findall(r"abstractvoice\[([a-z0-9,-]+)\]", text):
            named.update(group.split(","))
    missing = sorted(n for n in named if n not in extras)
    assert not missing, f"extras the installers use that pyproject.toml does not define: {missing}"
    # The CUDA step needs onnxruntime-gpu from the onnx-cuda extra.
    assert any(r.startswith("onnxruntime-gpu") for r in extras["onnx-cuda"])
    # Both scripts offer the same profiles (apple is refused on Windows, with a sentence).
    for p in ("auto", "light", "apple", "gpu"):
        assert f"'{p}'" in ps1, f"install.ps1 does not handle profile {p}"
        assert re.search(rf"^\s*{p}[|)]", sh, re.MULTILINE), f"install.sh does not handle profile {p}"


def test_scripts_and_docs_name_the_exact_one_liners():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    install = (ROOT / "docs" / "installation.md").read_text(encoding="utf-8")
    for doc, name in ((readme, "README.md"), (install, "docs/installation.md")):
        for line in (SH_LINE, PS1_LINE) + PIP_LINES:
            assert line in doc, f"{name} lacks: {line}"
    assert "## One-line install" in install
    assert SH_LINE in _sh()
    assert PS1_LINE in _ps1()
    assert "docs/installation.md#one-line-install" in _sh()
    assert "docs/installation.md#one-line-install" in _ps1()
