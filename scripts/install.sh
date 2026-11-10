#!/bin/sh
# =============================================================================
# AbstractVoice standalone installer (macOS / Linux)
# =============================================================================
# One line, no admin rights, no system Python needed:
#
#   curl -LsSf https://abstractframework.ai/voice/install.sh | sh
#   curl -LsSf https://abstractframework.ai/voice/install.sh | sh -s -- --profile gpu
#
# AbstractVoice alone (the abstractvoice and abstractvoice-prefetch commands, no gateway, no
# services); AbstractFramework's installer, https://abstractframework.ai/install.sh, includes it.
#
# Steps (each prints its command; --print shows them and changes nothing): preflight; uv
# (https://docs.astral.sh/uv) into ~/.local/bin when missing; Python 3.12 through uv; `uv tool
# install --python 3.12 "abstractvoice[<extras>]==<version>"` (user-scoped, prebuilt wheels only:
# no compiler, so the apple profile drops echo cancellation, which has no macOS wheel); the CUDA
# step on Linux with an NVIDIA card; `uv tool update-shell` when the command folder is not on PATH.
#
# Profiles (--profile auto|light|apple|gpu; auto takes the first that fits):
#   apple  Apple Silicon, macOS 14+: abstractvoice[apple] (MLX engines on the Apple GPU)
#   gpu    Linux with NVIDIA (nvidia-smi -L) or ROCm (rocminfo): abstractvoice[gpu] (torch
#          engines, Whisper on CUDA); asked for without a card, the same stack on the processor
#   light  anything else, processor only: Supertonic speech, faster-whisper transcription and
#          speaker/microphone I/O (abstractvoice[supertonic,stt,audio-io])
#
# The CUDA step (gpu profile, Linux, driver 580+ and compute capability 7.5+: onnxruntime-gpu 1.27+
# is CUDA 13). onnxruntime (CPU; piper-tts and faster-whisper require it) and onnxruntime-gpu write
# the same files; in one uv transaction (the extra, or --with onnxruntime-gpu) the CPU build's
# binding won 3 times out of 3 (Ubuntu 24.04, uv 0.13.7). The verified sequence:
#   uv tool install --python 3.12 "abstractvoice[gpu,onnx-cuda]==<version>"
#   uv pip install --python "$(uv tool dir)/abstractvoice/bin/python" --no-deps \
#       --reinstall-package onnxruntime-gpu "onnxruntime-gpu==<the version just installed>"
# The second rewrites onnxruntime-gpu's files alone, last, so the CUDA build wins; then
# `python -c "import onnxruntime as o; print(o.get_available_providers())"` lists
# CUDAExecutionProvider: proof the CUDA build is in place, not that its provider loads (Supertonic
# and Piper report that at first use). `uv tool upgrade` keeps the extra (it is in the tool's
# requirement); the two commands fit any uv tool environment carrying abstractvoice[onnx-cuda].
#
# Upgrade: run the same line again: "AbstractVoice <old> found: upgrading to <new>" (or "already
# up to date"); the previous profile is kept unless --profile says otherwise; the CUDA step runs
# again. Remove: uv tool uninstall abstractvoice
#
# Options (environment twins in brackets):
#   --profile auto|light|apple|gpu  install profile (default auto)              [AV_PROFILE]
#   --pin VERSION|latest     the AbstractVoice version (default: the release this script
#                            ships with, below; latest = the newest on PyPI)    [AV_PIN]
#   --from PATH              install from a checkout, sdist or wheel (testing)  [AV_FROM]
#   --no-modify-path         do not run `uv tool update-shell`
#   --print, --dry-run, -n   show the plan and the commands; change nothing
#   -v, --verbose            show the full output of every command
#   -h, --help               this help
# =============================================================================

if [ -n "${ZSH_VERSION:-}" ]; then emulate sh; fi
set -eu

# The release this script ships with = abstractvoice/_version.py (tests/test_install_scripts.py
# fails on drift), so an install lands on the version the script was tested with.
AV_PIN_DEFAULT="0.21.0"
AV_PYTHON="3.12"
AV_LIGHT_EXTRAS="supertonic,stt,audio-io"
AV_SCRIPT_URL="https://abstractframework.ai/voice/install.sh"
AV_DOCS="https://github.com/lpalbou/abstractvoice/blob/main/docs/installation.md#one-line-install"
AV_COMMANDS="abstractvoice abstractvoice-prefetch"

usage() {  # the header above when this runs as a file; a short form through curl | sh
    if [ -f "$0" ] && head -n 3 "$0" 2>/dev/null | grep -q "AbstractVoice standalone installer"; then
        awk 'NR > 4 && /^# =+$/ { exit } NR > 4 { sub(/^# ?/, ""); print }' "$0"
    else printf 'Usage: install.sh [--profile auto|light|apple|gpu] [--pin VERSION|latest] [--from PATH] [--print]\nFull help: %s\n' "$AV_DOCS"; fi
}
need_arg() { [ $# -ge 2 ] && [ -n "$2" ] || { echo "ERROR: $1 needs a value" >&2; exit 2; }; }

C_B=""; C_D=""; C_R=""; C_G=""; C_Y=""; C_C=""; C_0=""
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ] && [ "${TERM:-}" != "dumb" ]; then E="$(printf '\033')"
    C_B="$E[1m"; C_D="$E[2m"; C_R="$E[31m"; C_G="$E[32m"; C_Y="$E[33m"; C_C="$E[36m"; C_0="$E[0m"; fi
STEP=0; step() { STEP=$((STEP + 1)); printf '\n%s[%s] %s%s\n' "$C_B" "$STEP" "$1" "$C_0"; }
ok()   { printf '  %s✓%s %s\n' "$C_G" "$C_0" "$1"; }
info() { printf '  %s·%s %s\n' "$C_C" "$C_0" "$1"; }
warn() { printf '  %s!%s %s\n' "$C_Y" "$C_0" "$1"; }
die()  { printf '\n%sERROR: %s%s\n' "$C_R" "$1" "$C_0" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }
q() {  # shell-quote one word for display
    case "$1" in ''|*[!A-Za-z0-9_./:=@,+%-]*) printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")" ;; *) printf '%s' "$1" ;; esac
}
show_cmd() { _l=""; for _w in "$@"; do _l="$_l $(q "$_w")"; done; printf '%s' "${_l# }"; }

# run DESCRIPTION CMD...: print the command, then run it (output in the log; live as well with
# RUN_LIVE=1 or -v). RUN_SOFT=1: a failure warns and returns 1 instead of stopping the install.
LOG=""; RUN_RC=0; RUN_SOFT=0; RUN_LIVE=0; RUN_SHOW=""
run() {
    _desc="$1"; shift
    _shown="${RUN_SHOW:-$(show_cmd "$@")}"; RUN_SHOW=""
    _soft="$RUN_SOFT"; _live="$RUN_LIVE"; RUN_SOFT=0; RUN_LIVE=0; RUN_RC=0
    printf '  %s$ %s%s\n' "$C_D" "$_shown" "$C_0"
    [ "$PRINT" = 1 ] && return 0
    printf '\n$ %s\n' "$_shown" >>"$LOG"
    if [ "$VERBOSE" = 1 ] || [ "$_live" = 1 ]; then
        echo 0 >"$LOG.rc"
        { "$@" 2>&1 || echo "$?" >"$LOG.rc"; } | tee -a "$LOG"
        RUN_RC="$(cat "$LOG.rc")"
    else "$@" >>"$LOG" 2>&1 || RUN_RC=$?; fi
    [ "$RUN_RC" = 0 ] && return 0
    if [ "$_soft" = 1 ]; then warn "$_desc did not succeed (exit $RUN_RC; log: $LOG)"; return 1; fi
    tail -n 15 "$LOG" | sed 's/^/    /' >&2
    die "$_desc failed (exit $RUN_RC). Log: $LOG
What to do: run the installer again (it repairs a half-finished install); if it stops at the same step, report it with that log ($AV_DOCS)."
}
http() {  # http URL: the body on stdout; non-zero unless the answer is 2xx
    if have curl; then curl -fsS --connect-timeout 5 --max-time 20 "$1" 2>/dev/null; else wget -qO- --timeout=20 "$1" 2>/dev/null; fi
}
UV=""; find_uv() {
    if have uv; then UV="$(command -v uv)"; return 0; fi
    for _c in "${UV_INSTALL_DIR:-}/uv" "${XDG_BIN_HOME:-}/uv" "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv"; do
        if [ -x "$_c" ]; then UV="$_c"; return 0; fi; done; return 1
}
# The framework installer's nvidia_info: NV_OK (1 when nvidia-smi answers), NV_DRIVER, NV_MAJOR,
# NV_CC10 (the lowest compute capability x10, empty when the driver cannot report it), NV_NAME,
# NV_ERR (why an installed nvidia-smi gives no GPU; empty when there is no nvidia-smi at all).
nvidia_info() {
    NV_OK=0; NV_DRIVER=""; NV_MAJOR=0; NV_CC10=""; NV_NAME=""; NV_ERR=""
    have nvidia-smi || return 0
    nvidia-smi -L >/dev/null 2>&1 || { NV_ERR="nvidia-smi is installed but does not answer (driver not loaded?)"; return 0; }
    _nv="$(nvidia-smi --query-gpu=driver_version,compute_cap,name --format=csv,noheader 2>/dev/null)" || _nv=""
    if [ -n "$_nv" ]; then NV_NAME="$(printf '%s\n' "$_nv" | head -n 1 | cut -d, -f3- | sed 's/^ *//')"
        NV_CC10="$(printf '%s\n' "$_nv" | cut -d, -f2 | tr -d ' ' | awk -F. '$1 ~ /^[0-9]+$/ { v = $1 * 10 + ($2 == "" ? 0 : substr($2, 1, 1)); if (m == "" || v < m) m = v } END { print m }')"
    else  # drivers too old to report compute_cap: the driver alone
        _nv="$(nvidia-smi --query-gpu=driver_version,name --format=csv,noheader 2>/dev/null)" || _nv=""
        NV_NAME="$(printf '%s\n' "$_nv" | head -n 1 | cut -d, -f2- | sed 's/^ *//')"; fi
    NV_DRIVER="$(printf '%s\n' "$_nv" | head -n 1 | cut -d, -f1 | tr -d ' ')"; NV_MAJOR="${NV_DRIVER%%.*}"
    case "$NV_MAJOR" in ''|*[!0-9]*) NV_MAJOR=0; NV_ERR="nvidia-smi gave no driver version" ;; *) NV_OK=1 ;; esac
}

# The install is one function called on the last line, so a `curl | sh` cut mid-download runs
# nothing (the body is not indented, so multi-line messages stay flush).
main() {
PROFILE="${AV_PROFILE:-auto}"; PIN="${AV_PIN:-}"; FROM="${AV_FROM:-}"
NO_MODIFY_PATH=0; PRINT=0; VERBOSE=0
while [ $# -gt 0 ]; do
    case "$1" in
        --profile) need_arg "$@"; PROFILE="$2"; shift ;;  --profile=*) PROFILE="${1#*=}" ;;
        --pin) need_arg "$@"; PIN="$2"; shift ;;          --pin=*) PIN="${1#*=}" ;;
        --from) need_arg "$@"; FROM="$2"; shift ;;        --from=*) FROM="${1#*=}" ;;
        --no-modify-path) NO_MODIFY_PATH=1 ;;
        --print|--dry-run|-n) PRINT=1 ;;
        -v|--verbose) VERBOSE=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

# --- 1. Preflight -------------------------------------------------------------
OS="$(uname -s)"; ARCH="$(uname -m)"
case "$OS" in
    Darwin) OS_ID=macos ;;
    Linux)  OS_ID=linux ;;
    MINGW*|MSYS*|CYGWIN*) die "Windows: use powershell -ExecutionPolicy ByPass -c \"irm https://abstractframework.ai/voice/install.ps1 | iex\"" ;;
    *) die "unsupported OS: $OS (supported: macOS, Linux; Windows uses install.ps1)" ;;
esac
# A Terminal set to "Open using Rosetta" reports x86_64 on Apple Silicon: an Intel Python, no MLX.
if [ "$OS_ID" = macos ] && [ "$(sysctl -n sysctl.proc_translated 2>/dev/null || echo 0)" = 1 ]; then
    die "this Terminal runs in Intel (Rosetta) mode on an Apple Silicon Mac, so the installer would set up the
slow Intel version without the Apple Silicon engines. Run it natively: curl -LsSf $AV_SCRIPT_URL | arch -arm64 sh"
fi
MACOS_VERSION=""; MACOS_MAJOR=0; [ "$OS_ID" = macos ] && { MACOS_VERSION="$(sw_vers -productVersion 2>/dev/null || echo 0)"; MACOS_MAJOR="${MACOS_VERSION%%.*}"; }
MUSL=0; [ "$OS_ID" = linux ] && ldd --version 2>&1 | grep -qi musl && MUSL=1

printf '%sAbstractVoice installer%s  %s%s%s\n' "$C_B" "$C_0" "$C_D" \
    "$([ "$PRINT" = 1 ] && echo '(--print: preflight only, nothing is installed)' || echo "$AV_SCRIPT_URL")" "$C_0"
step "Preflight"
ok "system: $OS_ID $ARCH${MACOS_VERSION:+ (macOS $MACOS_VERSION)}"

find_uv || true
TOOL_BIN=""; TOOL_DIR=""
[ -z "$UV" ] || { TOOL_BIN="$("$UV" tool dir --bin 2>/dev/null || true)"; TOOL_DIR="$("$UV" tool dir 2>/dev/null || true)"; }
[ -n "$TOOL_BIN" ] || TOOL_BIN="${UV_TOOL_BIN_DIR:-${XDG_BIN_HOME:-$HOME/.local/bin}}"
[ -n "$TOOL_DIR" ] || TOOL_DIR="${UV_TOOL_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/uv/tools}"
BEFORE=""; PREV_PROFILE=""
if [ -n "$UV" ]; then  # the existing install: its version, and its profile from the uv receipt
    BEFORE="$("$UV" tool list 2>/dev/null | sed -n 's/^abstractvoice v\([^ ]*\).*/\1/p' | head -n 1)"
    _extras="$(sed -n 's/.*name = "abstractvoice", extras = \[\([^]]*\)\].*/\1/p' "$TOOL_DIR/abstractvoice/uv-receipt.toml" 2>/dev/null | head -n 1)"
    case "$_extras" in *'"apple"'*) PREV_PROFILE=apple ;; *'"gpu"'*) PREV_PROFILE=gpu ;; *'"supertonic"'*) PREV_PROFILE=light ;; esac
fi

nvidia_info; HAS_NVIDIA="$NV_OK"
HAS_ROCM=0; have rocminfo && rocminfo >/dev/null 2>&1 && HAS_ROCM=1
APPLE_OK=0; [ "$OS_ID" = macos ] && [ "$ARCH" = arm64 ] && [ "$MACOS_MAJOR" -ge 14 ] 2>/dev/null && APPLE_OK=1
case "$PROFILE" in
    auto|"")
        if [ -n "$PREV_PROFILE" ]; then PROFILE="$PREV_PROFILE"; _why="kept from the previous install"
        elif [ "$APPLE_OK" = 1 ]; then PROFILE=apple; _why="Apple Silicon, macOS $MACOS_VERSION"
        elif [ "$OS_ID" = macos ] && [ "$ARCH" = arm64 ]; then PROFILE=light
            _why="macOS $MACOS_VERSION: the Apple Silicon engines (MLX) need macOS 14 or later; update macOS and run the installer again"
        elif [ "$HAS_NVIDIA" = 1 ]; then PROFILE=gpu; _why="nvidia-smi found ${NV_NAME:-a GPU}"
        elif [ "$HAS_ROCM" = 1 ]; then PROFILE=gpu; _why="rocminfo found a GPU"
        elif [ -n "$NV_ERR" ]; then PROFILE=light; _why="$NV_ERR, so processor only; once the driver works, run the installer again"
        else PROFILE=light; _why="no GPU found, so processor only"; fi
        ok "profile: $PROFILE ($_why; override with --profile)" ;;
    light) ok "profile: light (processor only)" ;;
    apple)
        [ "$OS_ID" = macos ] && [ "$ARCH" = arm64 ] || die "the apple profile needs an Apple Silicon Mac (this is $OS_ID $ARCH); use --profile light"
        [ "$MACOS_MAJOR" -ge 14 ] 2>/dev/null || die "the apple profile needs macOS 14 or later (this is $MACOS_VERSION); use --profile light"
        ok "profile: apple (MLX engines on the Apple GPU)" ;;
    gpu)
        [ "$OS_ID" = linux ] || die "the gpu profile targets Linux (Windows: install.ps1); on a Mac use --profile apple"
        [ "$HAS_NVIDIA" = 1 ] || [ "$HAS_ROCM" = 1 ] || warn "gpu profile requested, but ${NV_ERR:-neither nvidia-smi nor rocminfo works}: its engines will run on the processor"
        ok "profile: gpu (CUDA/ROCm engines when a card works, else the processor)" ;;
    *) die "unknown profile '$PROFILE' (expected auto, light, apple or gpu)" ;;
esac
case "$PROFILE" in apple) EXTRAS=apple ;; gpu) EXTRAS=gpu ;; *) EXTRAS="$AV_LIGHT_EXTRAS" ;; esac
LOCAL_WHY=""
if [ "$PROFILE" = light ] && [ "$MUSL" = 1 ]; then LOCAL_WHY="ONNX Runtime and CTranslate2 publish no wheels for musl Linux (Alpine)"
elif [ "$PROFILE" = light ] && [ "$OS_ID" = macos ] && [ "$MACOS_MAJOR" -lt 13 ] 2>/dev/null; then LOCAL_WHY="ONNX Runtime publishes no wheels for macOS $MACOS_VERSION"; fi
[ -z "$LOCAL_WHY" ] || { EXTRAS=""; warn "local engines skipped: $LOCAL_WHY; the base package speaks through remote (OpenAI-compatible) audio"; }

# The CUDA step (see the top of this script).
CUDA=0; CUDA_RESULT="not applicable ($PROFILE profile)"
if [ "$PROFILE" = gpu ]; then
    _cpu="Supertonic and Piper run on the processor"
    if [ "$HAS_NVIDIA" = 0 ]; then CUDA_RESULT="skipped: ${NV_ERR:-no NVIDIA GPU answers nvidia-smi}; $_cpu"
    elif [ "$NV_MAJOR" -lt 580 ]; then CUDA_RESULT="skipped: onnxruntime-gpu is a CUDA 13 build, which needs NVIDIA driver 580 or newer (this one is $NV_DRIVER); $_cpu"
    elif [ -n "$NV_CC10" ] && [ "$NV_CC10" -lt 75 ]; then CUDA_RESULT="skipped: CUDA 13 needs compute capability 7.5 or newer; $_cpu"
    else CUDA=1; CUDA_RESULT="planned"; ok "NVIDIA: ${NV_NAME:-GPU}, driver $NV_DRIVER: ONNX Runtime's CUDA build is added (onnx-cuda)"; fi
    [ "$CUDA" = 1 ] || info "CUDA step $CUDA_RESULT"
fi

if [ -n "$FROM" ]; then
    [ -e "$FROM" ] || die "--from: no such file or folder: $FROM"
    FROM="$(cd "$(dirname "$FROM")" && pwd)/$(basename "$FROM")"; TARGET="(the build from $FROM)"; PIN_WHY="--from"
elif [ "$PIN" = latest ]; then
    _latest="$(http https://pypi.org/pypi/abstractvoice/json | grep -o '"version": *"[^"]*"' | head -n 1 | sed 's/.*"\([^"]*\)"$/\1/')" || _latest=""
    TARGET="${_latest:-the newest release on PyPI}"; PIN_WHY="--pin latest"
else
    if [ -n "$PIN" ]; then PIN_WHY="--pin"; else PIN="$AV_PIN_DEFAULT"; PIN_WHY="the release this installer ships with"; fi; TARGET="$PIN"
fi
spec() {  # spec EXTRAS: the requirement given to uv
    if [ -n "$FROM" ]; then printf 'abstractvoice%s @ file://%s' "${1:+[$1]}" "$FROM"
    elif [ "$PIN" = latest ]; then printf 'abstractvoice%s' "${1:+[$1]}"
    else printf 'abstractvoice%s==%s' "${1:+[$1]}" "$PIN"; fi
}
FULL_EXTRAS="$EXTRAS"; [ "$CUDA" = 0 ] || FULL_EXTRAS="$EXTRAS,onnx-cuda"; ok "package: $(spec "$FULL_EXTRAS")  ${C_D}(version: $PIN_WHY)${C_0}"
if [ -z "$BEFORE" ]; then ok "No AbstractVoice install found: installing AbstractVoice $TARGET"
elif [ -n "$FROM" ]; then ok "AbstractVoice $BEFORE found: replacing it with the build from $FROM (--from)"
elif [ "$BEFORE" = "$TARGET" ]; then ok "AbstractVoice $BEFORE found: already up to date (checked, repaired if needed)"
elif [ "$PIN" != latest ] && [ "$(printf '%s\n%s\n' "$BEFORE" "$PIN" | sort -V | tail -n 1)" = "$BEFORE" ]; then
    ok "AbstractVoice $BEFORE found: going back to AbstractVoice $TARGET (--pin)"
else ok "AbstractVoice $BEFORE found: upgrading to AbstractVoice $TARGET"; fi

# A command of the same name from another program (often AbstractFramework's gateway tool, which
# exposes abstractvoice itself): uv would refuse the install, so stop and say why.
if [ -z "$BEFORE" ]; then
    for _n in $AV_COMMANDS; do
        [ -e "$TOOL_BIN/$_n" ] || [ -L "$TOOL_BIN/$_n" ] || continue
        _owner="$([ -z "$UV" ] || "$UV" tool list 2>/dev/null | awk -v n="$_n" '/^[^ -]/ { t = $1 } $1 == "-" && $2 == n { print t; exit }')"
        _msg="$TOOL_BIN/$_n already exists ($([ -n "$_owner" ] && echo "from the uv tool '$_owner'" || echo "another program's"))"
        [ "$_owner" != abstractgateway ] || _msg="$_msg: AbstractFramework is installed here and already includes AbstractVoice; upgrade it with: curl -LsSf https://abstractframework.ai/install.sh | sh. A standalone AbstractVoice beside the framework's gateway tool is not supported (there is no --force: both would own the same commands)"
        [ "$PRINT" = 1 ] && { warn "$_msg (the real run stops here; nothing is changed)"; break; }
        die "$_msg
Nothing was changed."
    done
fi

case "$PROFILE" in apple) NEED_MB=6000 ;; gpu) NEED_MB=10000 ;; *) NEED_MB=1000 ;; esac
FREE_MB="$(df -Pk "$HOME" 2>/dev/null | awk 'NR == 2 { print int($4 / 1024) }')" || FREE_MB=""
if [ -z "$FREE_MB" ]; then :
elif [ "$FREE_MB" -lt 500 ]; then die "only $FREE_MB MB free in $HOME (about $NEED_MB MB needed)"
elif [ "$FREE_MB" -lt "$NEED_MB" ]; then warn "only $FREE_MB MB free in $HOME; the $PROFILE profile needs about $NEED_MB MB (models need more)"
else ok "disk: $FREE_MB MB free (about $NEED_MB MB needed, models extra)"; fi
if http https://pypi.org/simple/pip/ >/dev/null; then ok "internet: pypi.org reachable"
elif [ "$PRINT" = 1 ]; then warn "pypi.org is not reachable (a real run stops here)"
else die "this computer cannot reach pypi.org, where AbstractVoice is downloaded (check the connection or HTTPS_PROXY). Nothing was changed."; fi
if [ -z "$FROM" ] && [ "$PIN" != latest ] && ! http "https://pypi.org/pypi/abstractvoice/$PIN/json" >/dev/null; then
    _m="abstractvoice $PIN is not on PyPI (yet): use --pin latest for the newest published version"
    if [ "$PRINT" = 1 ]; then warn "$_m"; else die "$_m. Nothing was changed."; fi
fi
[ -n "$UV" ] || have curl || have wget || die "neither curl nor wget is installed, so uv cannot be downloaded"
WORK="${XDG_CACHE_HOME:-$HOME/.cache}/abstractvoice/installer"
[ "$PRINT" = 1 ] || { mkdir -p "$WORK"; LOG="$WORK/install.log"; : >"$LOG"; }

# --- 2. uv and Python ---------------------------------------------------------
step "uv (Python toolchain manager)"
if [ -n "$UV" ]; then ok "uv found: $UV ($("$UV" --version 2>/dev/null | awk '{print $2}'))"
else
    info "installing uv from astral.sh into ~/.local/bin (no admin)"
    if have curl; then run "download uv's installer" curl -LsSf https://astral.sh/uv/install.sh -o "$WORK/uv-install.sh"
    else run "download uv's installer" wget -qO "$WORK/uv-install.sh" https://astral.sh/uv/install.sh; fi
    run "install uv" env UV_NO_MODIFY_PATH=1 sh "$WORK/uv-install.sh"
    if [ "$PRINT" = 1 ]; then UV=uv
    else find_uv || die "uv was installed but cannot be found (looked in \$UV_INSTALL_DIR, \$XDG_BIN_HOME, ~/.local/bin)"
        ok "uv installed: $UV"; TOOL_BIN="$("$UV" tool dir --bin)"; TOOL_DIR="$("$UV" tool dir)"; fi
fi
step "Python $AV_PYTHON (managed by uv, isolated from any system Python)"
run "install Python $AV_PYTHON" "$UV" python install "$AV_PYTHON"

# --- 3. AbstractVoice ---------------------------------------------------------
step "AbstractVoice"
# Prebuilt wheels only: the apple profile asks for aec-audio-processing (echo cancellation) on
# macOS, which has no macOS wheel and would start a compiler (the Xcode tools dialog on a fresh
# Mac); an override whose marker is never true drops it, as in the framework installer. uv splits
# --overrides at whitespace, so the file is named relatively from its folder.
if [ "$PRINT" = 1 ]; then info "uv-overrides.txt: aec-audio-processing; sys_platform == 'never'"
else printf '%s\n' "aec-audio-processing; sys_platform == 'never'" >"$WORK/uv-overrides.txt"; fi
install_tool() {  # install_tool EXTRAS SOFT [uv options]
    _ex="$1"; _soft="$2"; shift 2
    set -- "$UV" tool install --python "$AV_PYTHON" --overrides uv-overrides.txt --no-build-package aec-audio-processing "$@"
    [ -z "$FROM" ] || set -- "$@" --reinstall-package abstractvoice  # same version number, new code
    # --pin latest: without --upgrade, uv keeps an installed version that still satisfies.
    [ "$PIN" != latest ] || [ -n "$FROM" ] || set -- "$@" --upgrade
    set -- "$@" "$(spec "$_ex")"
    RUN_SHOW="cd $(q "$WORK") && $(show_cmd "$@")"
    _cwd="$(pwd)"; [ "$PRINT" = 1 ] || cd "$WORK"
    _rc=0; RUN_LIVE=1 RUN_SOFT="$_soft" run "install $(spec "$_ex")" "$@" || _rc=1
    cd "$_cwd" 2>/dev/null || cd "$HOME"
    return "$_rc"
}
if [ "$CUDA" = 1 ]; then
    install_tool "$FULL_EXTRAS" 1 || {
        CUDA=0; CUDA_RESULT="not installed: abstractvoice[onnx-cuda] did not install here (log: $LOG); Supertonic and Piper run on the processor"
        warn "retrying without ONNX Runtime's CUDA build"; FULL_EXTRAS="$EXTRAS"; install_tool "$EXTRAS" 0; }
elif [ "$PROFILE" = light ] && [ -n "$EXTRAS" ]; then
    install_tool "$EXTRAS" 1 || {
        LOCAL_WHY="their packages did not install on this system (log: $LOG)"
        warn "the local engines (Supertonic, faster-whisper) did not install: retrying the base package"
        EXTRAS=""; FULL_EXTRAS=""; install_tool "" 0; }
else install_tool "$EXTRAS" 0; fi
# Repair: uv can report the tool installed while its command is missing or cannot start (an
# interrupted install) and then does nothing, so check the command and reinstall in place.
if [ "$PRINT" = 0 ] && ! "$TOOL_BIN/abstractvoice" --help >/dev/null 2>&1; then
    warn "the installed abstractvoice does not start (an earlier install was interrupted or damaged): reinstalling it"
    install_tool "$FULL_EXTRAS" 0 --reinstall
    "$TOOL_BIN/abstractvoice" --help >/dev/null 2>&1 || die "abstractvoice still does not start after reinstalling it (log: $LOG)"
fi
AFTER="$BEFORE"
if [ "$PRINT" = 0 ]; then
    AFTER="$("$UV" tool list 2>/dev/null | sed -n 's/^abstractvoice v\([^ ]*\).*/\1/p' | head -n 1)"
    if [ -z "$BEFORE" ]; then ok "installed abstractvoice $AFTER"
    elif [ "$BEFORE" != "$AFTER" ]; then ok "abstractvoice $BEFORE -> $AFTER"
    elif [ -n "$FROM" ]; then ok "abstractvoice $AFTER reinstalled from $FROM"
    else ok "abstractvoice $AFTER already installed"; fi
fi

# --- 4. The CUDA step (see the top of this script) -----------------------------
if [ "$CUDA" = 1 ]; then
    step "ONNX Runtime's CUDA build for Supertonic and Piper"
    TOOL_PY="$TOOL_DIR/abstractvoice/bin/python"; ORT_GPU="<the installed version>"
    [ "$PRINT" = 1 ] || ORT_GPU="$("$UV" pip freeze --python "$TOOL_PY" 2>/dev/null | sed -n 's/^onnxruntime-gpu==//p' | head -n 1)"
    _chk='import onnxruntime as o; print(",".join(o.get_available_providers()))'
    if [ -z "$ORT_GPU" ]; then CUDA_RESULT="not installed: onnxruntime-gpu is missing from the environment (log: $LOG)"; warn "$CUDA_RESULT"
    elif ! RUN_SOFT=1 run "write onnxruntime-gpu's files over the CPU build's" "$UV" pip install --python "$TOOL_PY" \
            --no-deps --reinstall-package onnxruntime-gpu "onnxruntime-gpu==$ORT_GPU"; then
        CUDA_RESULT="not installed: rewriting onnxruntime-gpu failed (log: $LOG); Supertonic and Piper may run on the processor"
    elif [ "$PRINT" = 1 ]; then
        printf '  %s$ %s%s\n' "$C_D" "$(show_cmd "$TOOL_PY" -c "$_chk")" "$C_0"; CUDA_RESULT="planned (onnxruntime-gpu, then the check that its CUDA build is in place)"
    else
        printf '  %s$ %s%s\n' "$C_D" "$(show_cmd "$TOOL_PY" -c "$_chk")" "$C_0"
        _prov="$("$TOOL_PY" -c "$_chk" 2>>"$LOG" | tail -n 1)" || _prov=""
        case ",$_prov," in
            *,CUDAExecutionProvider,*) CUDA_RESULT="onnxruntime-gpu $ORT_GPU: the CUDA build of ONNX Runtime is in place; Supertonic and Piper report the provider they run on at first use"; ok "$CUDA_RESULT" ;;
            *) CUDA_RESULT="onnxruntime-gpu $ORT_GPU installed but the CPU build's files are in place (providers: ${_prov:-none}; log: $LOG); Supertonic and Piper run on the processor"; warn "$CUDA_RESULT" ;;
        esac
    fi
fi

# --- 5. PATH and summary ------------------------------------------------------
step "PATH"
case ":$PATH:" in
    *":$TOOL_BIN:"*) ok "$TOOL_BIN is on PATH" ;;
    *)
        _hand="add it yourself: export PATH=\"$TOOL_BIN:\$PATH\" in your shell's startup file"
        # The startup files may name it as written here, as $HOME/..., ${HOME}/... or ~/..., or through
        # uv's `. "$HOME/.local/bin/env"` (which contains $HOME/.local/bin).
        _rel="${TOOL_BIN#"$HOME"}"; [ "$_rel" != "$TOOL_BIN" ] || _rel="//none"
        if grep -qsF -e "$TOOL_BIN" -e "\$HOME$_rel" -e "\${HOME}$_rel" -e "~$_rel" "$HOME/.zshenv" "$HOME/.zshrc" "$HOME/.bashrc" \
            "$HOME/.bash_profile" "$HOME/.profile" "${XDG_CONFIG_HOME:-$HOME/.config}/fish/conf.d/uv.env.fish"; then
            ok "$TOOL_BIN is already on PATH in your shell profile (new terminals have it)"
        elif [ "$NO_MODIFY_PATH" = 1 ]; then warn "$TOOL_BIN is not on PATH (--no-modify-path); $_hand"
        elif RUN_SOFT=1 run "add $TOOL_BIN to PATH in your shell profile" "$UV" tool update-shell; then
            info "open a new terminal for the 'abstractvoice' command to be on PATH"
        else info "$_hand"; fi ;;
esac

printf '\n%s%s%s\n' "$C_B" "$([ "$PRINT" = 1 ] && echo 'Plan (--print: nothing was changed)' || echo 'AbstractVoice is installed')" "$C_0"
_ver="${AFTER:-$TARGET}"; [ "$PRINT" = 1 ] && _ver="$TARGET"
echo "  version:   $_ver$([ "$PRINT" = 0 ] && [ -n "$BEFORE" ] && [ "$BEFORE" != "$AFTER" ] && echo " (was $BEFORE)")"
echo "  profile:   $PROFILE ($(spec "$FULL_EXTRAS"))"
echo "  commands:  $TOOL_BIN/abstractvoice, $TOOL_BIN/abstractvoice-prefetch"
echo "  CUDA step: $CUDA_RESULT"
[ -z "$LOCAL_WHY" ] || echo "  local engines: skipped ($LOCAL_WHY)"
[ "$OS_ID" != linux ] || ! have ldconfig || ldconfig -p 2>/dev/null | grep -q libportaudio \
    || echo "  speakers/microphone: need PortAudio (Debian/Ubuntu: sudo apt-get install libportaudio2); audio files work without it"
[ "$PRINT" = 1 ] || echo "  log:       $LOG"
echo
if [ -n "$EXTRAS" ]; then echo "Try it (the first command downloads the Supertonic voice once, about 380 MB):"
    echo "  abstractvoice-prefetch --supertonic"; echo "  abstractvoice --tts-engine supertonic --prompt \"Hello from AbstractVoice.\" --output hello.wav"
else echo "Try it: abstractvoice --help"; fi
echo "Docs: $AV_DOCS"
}

main "$@"
