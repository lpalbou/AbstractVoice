<#
.SYNOPSIS
    AbstractVoice standalone installer for Windows 10/11 (PowerShell 5.1 and 7).

.DESCRIPTION
    One line, no admin rights, no system Python needed:

        powershell -ExecutionPolicy ByPass -c "irm https://abstractframework.ai/voice/install.ps1 | iex"

    With options (a script block keeps the parameters):
        & ([scriptblock]::Create((irm https://abstractframework.ai/voice/install.ps1))) -Profile gpu

    The Windows twin of scripts/install.sh (its header has the reasoning): AbstractVoice alone, no
    gateway, no services. Each step prints its command; -Print shows them and changes nothing.
    Steps: preflight; uv into %USERPROFILE%\.local\bin when missing; Python 3.12 through uv; uv tool
    install --python 3.12 "abstractvoice[<extras>]==<version>" (prebuilt wheels only, no MSVC Build
    Tools); the CUDA step with an NVIDIA card; `uv tool update-shell` for the user PATH.

    Profiles (-Profile auto|light|gpu; auto takes the first that fits):
      gpu    nvidia-smi answers: abstractvoice[gpu] with PyTorch's CUDA build for the driver
             (580+ and compute capability 7.5+: --torch-backend cu130; 525+: cu126; else PyPI's
             CPU build). Asked for without a card: the same stack, on the processor.
      light  processor only: Supertonic speech, faster-whisper transcription, speaker/microphone
             I/O (abstractvoice[supertonic,stt,audio-io]); Windows on ARM: abstractvoice[supertonic]
             (CTranslate2 and webrtcvad-wheels publish no ARM64 wheels). apple: refused here.

    The CUDA step (driver 580+ and compute capability 7.5+: onnxruntime-gpu 1.27+ is CUDA 13).
    onnxruntime (CPU; piper-tts and faster-whisper require it) and onnxruntime-gpu write the same
    files and in one uv transaction the CPU build won, so (verified on Linux by install.sh):
        uv tool install --python 3.12 "abstractvoice[gpu,onnx-cuda]==<version>"
        uv pip install --python "$(uv tool dir)\abstractvoice\Scripts\python.exe" --no-deps `
            --reinstall-package onnxruntime-gpu "onnxruntime-gpu==<the version just installed>"
    then o.get_available_providers() must list CUDAExecutionProvider: proof the CUDA build is in
    place, not that its provider loads (Supertonic and Piper report that at first use).

    Upgrade: run the same line again ("AbstractVoice <old> found: upgrading to <new>"); the profile
    is kept unless -Profile says otherwise. Remove: uv tool uninstall abstractvoice

    Options (environment twins AV_PROFILE, AV_PIN, AV_FROM): -Profile auto|light|gpu;
    -Pin VERSION|latest (default: the release this script ships with; latest = the newest on PyPI);
    -From PATH (a checkout, sdist or wheel, for testing); -NoModifyPath; -Print (alias -DryRun);
    -Verbose (every command's full output). The body runs only from Main, called on the last line.

.EXAMPLE
    .\install.ps1 -Print
#>
[CmdletBinding()]
param(
    # No [ValidateSet]: `irm | iex` evaluates this block as statements; Main validates the value.
    [Alias('Profile')]
    [string]$InstallProfile = '',
    [string]$Pin = '',
    [string]$From = '',
    [switch]$NoModifyPath,
    [Alias('DryRun')]
    [switch]$Print
)

# The release this script ships with = abstractvoice/_version.py (tests/test_install_scripts.py).
$AvPinDefault = '0.21.1'
$AvPython = '3.12'; $AvCommands = @('abstractvoice', 'abstractvoice-prefetch')
$AvLightExtras = 'supertonic,stt,audio-io'
$AvLightExtrasArm64 = 'supertonic'
$AvScriptUrl = 'https://abstractframework.ai/voice/install.ps1'
$AvDocs = 'https://github.com/lpalbou/abstractvoice/blob/main/docs/installation.md#one-line-install'

$script:RanAsFile = [bool]$PSCommandPath; $script:DryRun = [bool]$Print; $script:LogFile = ''; $script:StepNo = 0
$script:ShowAll = ($VerbosePreference -eq 'Continue')   # -Verbose (CmdletBinding): show every command's output

function Write-Step([string]$Text) { $script:StepNo++; Write-Host ''; Write-Host "[$($script:StepNo)] $Text" -ForegroundColor White }
function Write-Mark([string]$Mark, [string]$Color, [string]$Text) { Write-Host "  $Mark " -ForegroundColor $Color -NoNewline; Write-Host $Text }
function Write-Ok([string]$Text) { Write-Mark '+' Green $Text }; function Write-Info([string]$Text) { Write-Mark '.' Cyan $Text }
function Write-Warn2([string]$Text) { Write-Mark '!' Yellow $Text }
function Stop-Install([string]$Text) { throw "AVBOOT: $Text" }
function Test-Command([string]$Name) { return [bool](Get-Command $Name -ErrorAction SilentlyContinue) }
function Format-Arg([string]$Value) { if ($Value -match '^[A-Za-z0-9_./:=@,+%\\-]+$') { return $Value }; return "'" + ($Value -replace "'", "''") + "'" }
function Format-Cmd([string[]]$Argv) { return (($Argv | ForEach-Object { Format-Arg $_ }) -join ' ') }
function Get-Http([string]$Url) { try { return [string](Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 20 -ErrorAction Stop -Verbose:$false).Content } catch { return $null } }

# Run a native command: print it, log its output (and show it with -Live), return $true on exit 0.
# A failure stops the install unless -Soft, which warns and returns $false.
function Invoke-Native {
    param([string]$Description, [string[]]$Argv, [switch]$Soft, [switch]$Live, [string]$Shown = '')
    if (-not $Shown) { $Shown = Format-Cmd $Argv }
    Write-Host "  `$ $Shown" -ForegroundColor DarkGray
    if ($script:DryRun) { return $true }
    $exe = $Argv[0]; $rest = @(); if ($Argv.Count -gt 1) { $rest = $Argv[1..($Argv.Count - 1)] }
    $old = $ErrorActionPreference; $ErrorActionPreference = 'Continue'   # PS 5.1 turns native stderr into errors
    try {
        Add-Content -Path $script:LogFile -Value "`r`n`$ $Shown" -Encoding UTF8
        & $exe @rest 2>&1 | ForEach-Object {
            $line = "$_"; Add-Content -Path $script:LogFile -Value $line -Encoding UTF8
            if ($Live -or $script:ShowAll) { Write-Host "    $line" -ForegroundColor DarkGray }
        }
        $code = $LASTEXITCODE
    } catch { Add-Content -Path $script:LogFile -Value "$_" -Encoding UTF8; $code = 1 } finally { $ErrorActionPreference = $old }
    if ($code -eq 0) { return $true }
    if ($Soft) { Write-Warn2 "$Description did not succeed (exit $code; log: $($script:LogFile))"; return $false }
    Get-Content -Path $script:LogFile -Tail 15 | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkGray }
    Stop-Install "$Description failed (exit $code). Log: $($script:LogFile)`nWhat to do: run the installer again (it repairs a half-finished install); if it stops at the same step, report it with that log ($AvDocs)."
}

# The framework installer's Get-NvidiaInfo: Ok, Driver, DriverMajor, Cc (the lowest compute
# capability, $null when the driver cannot report it), Name, Error (why an installed nvidia-smi
# gives no GPU; empty when there is no nvidia-smi at all).
function Get-NvidiaInfo {
    $info = @{ Ok = $false; Driver = ''; DriverMajor = 0; Cc = $null; Name = ''; Error = '' }
    $noAnswer = 'nvidia-smi is installed but does not answer (driver not loaded?)'
    if (-not (Test-Command 'nvidia-smi')) { return $info }
    $old = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
    try {
        $withCc = $true
        $lines = @(& nvidia-smi '--query-gpu=driver_version,compute_cap,name' '--format=csv,noheader' 2>$null | ForEach-Object { "$_" })
        if ($LASTEXITCODE -ne 0) {   # drivers too old to report compute_cap: the driver alone
            $withCc = $false; $lines = @(& nvidia-smi '--query-gpu=driver_version,name' '--format=csv,noheader' 2>$null | ForEach-Object { "$_" }) }
        if ($LASTEXITCODE -ne 0) { $info.Error = $noAnswer; return $info }
    } catch { $info.Error = $noAnswer; return $info } finally { $ErrorActionPreference = $old }
    $inv = [Globalization.CultureInfo]::InvariantCulture; $nameAt = if ($withCc) { 2 } else { 1 }
    foreach ($l in ($lines | Where-Object { $_.Trim() })) {
        $parts = $l.Split(',', $nameAt + 1); $major = 0; $cc = 0.0
        if (-not [int]::TryParse(($parts[0].Trim().Split('.')[0]), [ref]$major)) { continue }
        if (-not $info.Driver) { $info.Driver = $parts[0].Trim(); $info.DriverMajor = $major }
        if ($withCc -and $parts.Count -ge 2 -and [double]::TryParse($parts[1].Trim(), [Globalization.NumberStyles]::Float, $inv, [ref]$cc)) {
            if ($null -eq $info.Cc -or $cc -lt $info.Cc) { $info.Cc = $cc }
        }
        if (-not $info.Name -and $parts.Count -gt $nameAt) { $info.Name = $parts[$nameAt].Trim() }
    }
    $info.Ok = [bool]$info.Driver; if (-not $info.Ok) { $info.Error = 'nvidia-smi gave no driver version' }; return $info
}

function Find-Uv([string]$HomeDir) {
    $cmd = Get-Command uv -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    foreach ($d in @($env:UV_INSTALL_DIR, $env:XDG_BIN_HOME, [IO.Path]::Combine($HomeDir, '.local', 'bin'), [IO.Path]::Combine($HomeDir, '.cargo', 'bin'))) {
        if ($d -and (Test-Path -LiteralPath (Join-Path $d 'uv.exe'))) { return (Join-Path $d 'uv.exe') }
    }
    return ''
}
function Get-ToolList([string]$Uv) {  # `uv tool list` lines (none without uv)
    if (-not $Uv) { return @() }
    $old = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
    try { return @(& $Uv tool list 2>$null | ForEach-Object { "$_" }) } catch { return @() } finally { $ErrorActionPreference = $old }
}
function Get-ToolVersion([string]$Uv) { foreach ($l in (Get-ToolList $Uv)) { if ($l -match '^abstractvoice v(\S+)') { return $Matches[1] } }; return '' }

function Main {
    $onWindows = ($PSVersionTable.PSEdition -eq 'Desktop') -or ($IsWindows -eq $true)
    if (-not $onWindows -and -not $script:DryRun) {
        Stop-Install 'this installer is for Windows; on macOS and Linux run: curl -LsSf https://abstractframework.ai/voice/install.sh | sh'
    }
    $homeDir = if ($env:USERPROFILE) { $env:USERPROFILE } else { $HOME }
    $profileName = $InstallProfile
    if (-not $profileName) { $profileName = if ($env:AV_PROFILE) { $env:AV_PROFILE } else { 'auto' } }
    if (-not $Pin -and $env:AV_PIN) { $Pin = $env:AV_PIN }
    if (-not $From -and $env:AV_FROM) { $From = $env:AV_FROM }
    $arch = if ($env:PROCESSOR_ARCHITEW6432) { $env:PROCESSOR_ARCHITEW6432 } else { $env:PROCESSOR_ARCHITECTURE }

    Write-Host "AbstractVoice installer  $(if ($script:DryRun) { '(-Print: preflight only, nothing is installed)' } else { $AvScriptUrl })" -ForegroundColor White
    Write-Step 'Preflight'
    Write-Ok "system: Windows $([Environment]::OSVersion.Version) $arch"

    $uv = Find-Uv $homeDir; $toolBin = ''; $toolDir = ''
    if ($uv) { $toolBin = (& $uv tool dir --bin 2>$null | Select-Object -First 1); $toolDir = (& $uv tool dir 2>$null | Select-Object -First 1) }
    if (-not $toolBin) { $toolBin = if ($env:UV_TOOL_BIN_DIR) { $env:UV_TOOL_BIN_DIR } else { [IO.Path]::Combine($homeDir, '.local', 'bin') } }
    if (-not $toolDir) { $toolDir = if ($env:UV_TOOL_DIR) { $env:UV_TOOL_DIR } else { [IO.Path]::Combine($(if ($env:APPDATA) { $env:APPDATA } else { $homeDir }), 'uv', 'tools') } }
    $before = Get-ToolVersion $uv; $prevProfile = ''   # the existing install: its version, and its profile from the uv receipt
    $receipt = [IO.Path]::Combine($toolDir, 'abstractvoice', 'uv-receipt.toml')
    if ($before -and (Test-Path -LiteralPath $receipt)) {
        $rc = Get-Content -LiteralPath $receipt -Raw
        if ($rc -match 'name = "abstractvoice", extras = \[[^\]]*"gpu"') { $prevProfile = 'gpu' } elseif ($rc -match 'name = "abstractvoice", extras = \[[^\]]*"supertonic"') { $prevProfile = 'light' } }

    $gpu = Get-NvidiaInfo
    switch ($profileName) {
        'auto' {
            if ($prevProfile) { $profileName = $prevProfile; $why = 'kept from the previous install' }
            elseif ($gpu.Ok) { $profileName = 'gpu'; $why = "nvidia-smi found $(if ($gpu.Name) { $gpu.Name } else { 'a GPU' })" }
            elseif ($gpu.Error) { $profileName = 'light'; $why = "$($gpu.Error), so processor only; once the driver works, run the installer again" }
            else { $profileName = 'light'; $why = 'no GPU found, so processor only' }
            Write-Ok "profile: $profileName ($why; override with -Profile)"
        }
        'light' { Write-Ok 'profile: light (processor only)' }
        'gpu' {
            if ($arch -eq 'ARM64') { Stop-Install 'the gpu profile has no Windows on ARM build (PyTorch CUDA, CTranslate2 and webrtcvad-wheels publish no ARM64 wheels); use -Profile light' }
            if (-not $gpu.Ok) { Write-Warn2 "gpu profile requested, but $(if ($gpu.Error) { $gpu.Error } else { 'there is no nvidia-smi' }): its engines will run on the processor" }
            Write-Ok 'profile: gpu (CUDA engines when a card works, else the processor)'
        }
        'apple' { Stop-Install 'the apple profile is for Apple Silicon Macs; use -Profile light or gpu' }
        default { Stop-Install "unknown profile '$profileName' (expected auto, light or gpu)" }
    }
    $extras = if ($profileName -eq 'gpu') { 'gpu' } elseif ($arch -eq 'ARM64') { $AvLightExtrasArm64 } else { $AvLightExtras }
    if ($profileName -eq 'light' -and $arch -eq 'ARM64') { Write-Info 'Windows on ARM: Supertonic only (CTranslate2 and webrtcvad-wheels publish no ARM64 wheels)' }

    # PyTorch's build (gpu profile; the framework installer's Select-GpuStack, without llama.cpp).
    $torchArgs = @()
    if ($profileName -eq 'gpu') {
        $torch = ''
        if ($gpu.Ok -and $gpu.DriverMajor -ge 580 -and $null -ne $gpu.Cc -and $gpu.Cc -ge 7.5) { $torch = 'cu130' }
        elseif ($gpu.Ok -and $gpu.DriverMajor -ge 525) { $torch = 'cu126' }
        if ($torch) {
            $help = if ($uv) { (& $uv tool install --help 2>$null) -join "`n" } else { '--torch-backend' }
            if ($help -match '--torch-backend') { $torchArgs = @('--torch-backend', $torch); Write-Ok "PyTorch: CUDA build $torch (driver $($gpu.Driver))" }
            else { Write-Warn2 "this uv has no --torch-backend: PyTorch's CPU build is installed; update uv (uv self update) and run the installer again" }
        } else { Write-Info "PyTorch: PyPI's CPU build ($(if ($gpu.Ok) { "driver $($gpu.Driver) is older than 525" } else { 'no NVIDIA GPU' }))" }
    }

    # The CUDA step (see the top of this script).
    $cuda = $false; $cpu = 'Supertonic and Piper run on the processor'; $cudaResult = "not applicable ($profileName profile)"
    if ($profileName -eq 'gpu') {
        if (-not $gpu.Ok) { $cudaResult = "skipped: $(if ($gpu.Error) { $gpu.Error } else { 'no NVIDIA GPU answers nvidia-smi' }); $cpu" }
        elseif ($gpu.DriverMajor -lt 580) { $cudaResult = "skipped: onnxruntime-gpu is a CUDA 13 build, which needs NVIDIA driver 580 or newer (this one is $($gpu.Driver)); $cpu" }
        elseif ($null -ne $gpu.Cc -and $gpu.Cc -lt 7.5) { $cudaResult = "skipped: CUDA 13 needs compute capability 7.5 or newer; $cpu" }
        else { $cuda = $true; $cudaResult = 'planned'; Write-Ok "NVIDIA: $($gpu.Name), driver $($gpu.Driver): ONNX Runtime's CUDA build is added (onnx-cuda)" }
        if (-not $cuda) { Write-Info "CUDA step $cudaResult" }
    }

    if ($From) {
        if (-not (Test-Path -LiteralPath $From)) { Stop-Install "-From: no such file or folder: $From" }
        $From = (Resolve-Path -LiteralPath $From).Path; $target = "(the build from $From)"; $pinWhy = '-From'
    } elseif ($Pin -eq 'latest') {
        $latest = ''; $j = Get-Http 'https://pypi.org/pypi/abstractvoice/json'
        if ($j -and $j -match '"version":\s*"([^"]+)"') { $latest = $Matches[1] }
        $target = if ($latest) { $latest } else { 'the newest release on PyPI' }; $pinWhy = '-Pin latest'
    } else {
        if ($Pin) { $pinWhy = '-Pin' } else { $Pin = $AvPinDefault; $pinWhy = 'the release this installer ships with' }; $target = $Pin
    }
    function Get-Spec([string]$Extras) {  # the requirement given to uv
        $x = if ($Extras) { "[$Extras]" } else { '' }
        if ($From) { return "abstractvoice$x @ " + [Uri]::new($From, [UriKind]::Absolute).AbsoluteUri }
        if ($Pin -eq 'latest') { return "abstractvoice$x" }; return "abstractvoice$x==$Pin"
    }
    $fullExtras = if ($cuda) { "$extras,onnx-cuda" } else { $extras }
    Write-Ok "package: $(Get-Spec $fullExtras)  (version: $pinWhy)"
    if (-not $before) { Write-Ok "No AbstractVoice install found: installing AbstractVoice $target" }
    elseif ($From) { Write-Ok "AbstractVoice $before found: replacing it with the build from $From (-From)" }
    elseif ($before -eq $target) { Write-Ok "AbstractVoice $before found: already up to date (checked, repaired if needed)" }
    elseif ($Pin -ne 'latest' -and $(try { [version]($before -replace '[^0-9.].*$', '') -gt [version]($Pin -replace '[^0-9.].*$', '') } catch { $false })) {
        Write-Ok "AbstractVoice $before found: going back to AbstractVoice $target (-Pin)"
    } else { Write-Ok "AbstractVoice $before found: upgrading to AbstractVoice $target" }

    # A command of the same name from another program (often AbstractFramework's gateway tool):
    # uv would refuse the install, so stop and say why.
    if (-not $before) {
        foreach ($n in $AvCommands) {
            $exe = Join-Path $toolBin "$n.exe"
            if (-not (Test-Path -LiteralPath $exe)) { continue }
            $owner = ''; $t = ''
            foreach ($l in (Get-ToolList $uv)) { if ($l -match '^(\S+) v') { $t = $Matches[1] } elseif ($l -eq "- $n") { $owner = $t; break } }
            $msg = "$exe already exists ($(if ($owner) { "from the uv tool '$owner'" } else { "another program's" }))"
            if ($owner -eq 'abstractgateway') { $msg += ': AbstractFramework is installed here and already includes AbstractVoice; upgrade it with: powershell -ExecutionPolicy ByPass -c "irm https://abstractframework.ai/install.ps1 | iex". A standalone AbstractVoice beside the framework''s gateway tool is not supported (there is no -Force: both would own the same commands)' }
            if ($script:DryRun) { Write-Warn2 "$msg (the real run stops here; nothing is changed)"; break }
            Stop-Install "$msg`nNothing was changed."
        }
    }

    $needMb = if ($profileName -eq 'gpu') { 10000 } else { 1000 }
    try {
        $root = [System.IO.Path]::GetPathRoot($homeDir)
        $freeMb = [int]((New-Object System.IO.DriveInfo($root)).AvailableFreeSpace / 1MB)
        if ($freeMb -lt 500) { Stop-Install "only $freeMb MB free on $root (about $needMb MB needed)" }
        elseif ($freeMb -lt $needMb) { Write-Warn2 "only $freeMb MB free on $root; the $profileName profile needs about $needMb MB (models need more)" }
        else { Write-Ok "disk: $freeMb MB free (about $needMb MB needed, models extra)" }
    } catch { if ("$_" -like 'AVBOOT:*') { throw } }
    if ($null -ne (Get-Http 'https://pypi.org/simple/pip/')) { Write-Ok 'internet: pypi.org reachable' }
    elseif ($script:DryRun) { Write-Warn2 'pypi.org is not reachable (a real run stops here)' }
    else { Stop-Install 'this computer cannot reach pypi.org, where AbstractVoice is downloaded (check the connection or the proxy). Nothing was changed.' }
    if (-not $From -and $Pin -ne 'latest' -and $null -eq (Get-Http "https://pypi.org/pypi/abstractvoice/$Pin/json")) {
        $m = "abstractvoice $Pin is not on PyPI (yet): use -Pin latest for the newest published version"
        if ($script:DryRun) { Write-Warn2 $m } else { Stop-Install "$m. Nothing was changed." }
    }
    $work = [IO.Path]::Combine($(if ($env:LOCALAPPDATA) { $env:LOCALAPPDATA } else { $homeDir }), 'abstractvoice', 'installer')
    if (-not $script:DryRun) { New-Item -ItemType Directory -Force -Path $work | Out-Null
        $script:LogFile = Join-Path $work 'install.log'; Set-Content -Path $script:LogFile -Value '' -Encoding UTF8 }

    # --- 2. uv and Python ------------------------------------------------------------------------
    Write-Step 'uv (Python toolchain manager)'
    if ($uv) { Write-Ok "uv found: $uv" }
    else {
        Write-Info 'installing uv from astral.sh into %USERPROFILE%\.local\bin (no admin)'
        # A child PowerShell with a per-process ByPass policy, so the vendor script's `exit` cannot end this one.
        $command = "`$env:UV_NO_MODIFY_PATH='1'; irm https://astral.sh/uv/install.ps1 | iex"
        Invoke-Native -Description 'install uv' -Argv @((Get-Process -Id $PID).Path, '-NoProfile', '-ExecutionPolicy', 'ByPass', '-Command', $command) `
            -Shown "powershell -ExecutionPolicy ByPass -c `"$command`"" | Out-Null
        if ($script:DryRun) { $uv = 'uv' }
        else { $uv = Find-Uv $homeDir; if (-not $uv) { Stop-Install 'uv was installed but cannot be found in %USERPROFILE%\.local\bin' }
            Write-Ok "uv installed: $uv"
            $toolBin = (& $uv tool dir --bin 2>$null | Select-Object -First 1); $toolDir = (& $uv tool dir 2>$null | Select-Object -First 1) }
    }
    Write-Step "Python $AvPython (managed by uv, isolated from any system Python)"
    Invoke-Native -Description "install Python $AvPython" -Argv @($uv, 'python', 'install', $AvPython) | Out-Null

    # --- 3. AbstractVoice ------------------------------------------------------------------------
    # Prebuilt wheels only: every package of these profiles has a Windows wheel for Python 3.12
    # (aec-audio-processing too), so no override is needed here (install.sh drops it on macOS).
    Write-Step 'AbstractVoice'
    function Install-Tool([string]$Extras, [switch]$Soft, [string[]]$More = @()) {
        $argv = @($uv, 'tool', 'install', '--python', $AvPython) + $More
        if ($Extras -like 'gpu*') { $argv += $torchArgs }
        if ($From) { $argv += @('--reinstall-package', 'abstractvoice') }       # same version number, new code
        elseif ($Pin -eq 'latest') { $argv += '--upgrade' }                     # else uv keeps a satisfying install
        $argv += (Get-Spec $Extras)
        return (Invoke-Native -Description "install $(Get-Spec $Extras)" -Argv $argv -Live -Soft:$Soft)
    }
    $localWhy = ''
    if ($cuda) {
        if (-not (Install-Tool $fullExtras -Soft)) {
            $cuda = $false; $cudaResult = "not installed: abstractvoice[onnx-cuda] did not install here (log: $($script:LogFile)); $cpu"
            Write-Warn2 "retrying without ONNX Runtime's CUDA build"; $fullExtras = $extras; Install-Tool $extras | Out-Null
        }
    } elseif ($profileName -eq 'light') {
        if (-not (Install-Tool $extras -Soft)) {
            $localWhy = "their packages did not install on this system (log: $($script:LogFile))"
            Write-Warn2 'the local engines (Supertonic, faster-whisper) did not install: retrying the base package'
            $extras = ''; $fullExtras = ''; Install-Tool '' | Out-Null
        }
    } else { Install-Tool $extras | Out-Null }
    $avExe = Join-Path $toolBin 'abstractvoice.exe'
    $after = $before
    if (-not $script:DryRun) {
        # Repair: uv can report the tool installed while its command cannot start; reinstall in place.
        & $avExe --help *> $null
        if ($LASTEXITCODE -ne 0) {
            Write-Warn2 'the installed abstractvoice does not start (an earlier install was interrupted or damaged): reinstalling it'
            Install-Tool $fullExtras -More @('--reinstall') | Out-Null
            & $avExe --help *> $null
            if ($LASTEXITCODE -ne 0) { Stop-Install "abstractvoice still does not start after reinstalling it (log: $($script:LogFile))" }
        }
        $after = Get-ToolVersion $uv
        if (-not $before) { Write-Ok "installed abstractvoice $after" }
        elseif ($before -ne $after) { Write-Ok "abstractvoice $before -> $after" }
        elseif ($From) { Write-Ok "abstractvoice $after reinstalled from $From" }
        else { Write-Ok "abstractvoice $after already installed" }
    }

    # --- 4. The CUDA step --------------------------------------------------------------------------
    if ($cuda) {
        Write-Step "ONNX Runtime's CUDA build for Supertonic and Piper"
        $toolPy = [IO.Path]::Combine($toolDir, 'abstractvoice', 'Scripts', 'python.exe')
        $ortGpu = if ($script:DryRun) { '<the installed version>' } else { '' }
        if (-not $script:DryRun) { foreach ($l in @(& $uv pip freeze --python $toolPy 2>$null | ForEach-Object { "$_" })) { if ($l -match '^onnxruntime-gpu==(\S+)') { $ortGpu = $Matches[1] } } }
        # From a file, never `python -c`: PowerShell 5.1 strips inner double quotes from native arguments (and the code has none).
        $chk = "import onnxruntime as o`nprint(','.join(o.get_available_providers()))`n"
        $chkFile = [IO.Path]::Combine($work, 'provider-check.py')
        if (-not $ortGpu) { $cudaResult = "not installed: onnxruntime-gpu is missing from the environment (log: $($script:LogFile))"; Write-Warn2 $cudaResult }
        elseif (-not (Invoke-Native -Description "write onnxruntime-gpu's files over the CPU build's" -Soft `
                -Argv @($uv, 'pip', 'install', '--python', $toolPy, '--no-deps', '--reinstall-package', 'onnxruntime-gpu', "onnxruntime-gpu==$ortGpu"))) {
            $cudaResult = "not installed: rewriting onnxruntime-gpu failed (log: $($script:LogFile)); Supertonic and Piper may run on the processor"
        } else {
            Write-Host "  `$ $(Format-Cmd @($toolPy, $chkFile))   # $($chk.Trim() -replace "`n", '; ')" -ForegroundColor DarkGray
            if ($script:DryRun) { $cudaResult = 'planned (onnxruntime-gpu, then the check that its CUDA build is in place)' }
            else {
                [IO.File]::WriteAllText($chkFile, $chk, (New-Object System.Text.UTF8Encoding($false)))
                $old = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
                try { $prov = "$(& $toolPy $chkFile 2>$null | Select-Object -Last 1)" } catch { $prov = '' } finally { $ErrorActionPreference = $old }
                if ((",$prov,") -like '*,CUDAExecutionProvider,*') { $cudaResult = "onnxruntime-gpu ${ortGpu}: the CUDA build of ONNX Runtime is in place; Supertonic and Piper report the provider they run on at first use"; Write-Ok $cudaResult }
                else { $cudaResult = "onnxruntime-gpu $ortGpu installed but the CPU build's files are in place (providers: $(if ($prov) { $prov } else { 'none' }); log: $($script:LogFile)); $cpu"; Write-Warn2 $cudaResult }
            }
        }
    }

    # --- 5. PATH and summary -----------------------------------------------------------------------
    Write-Step 'PATH'
    if (($env:PATH -split [IO.Path]::PathSeparator) -contains $toolBin) { Write-Ok "$toolBin is on PATH" }
    elseif ($NoModifyPath) { Write-Warn2 "$toolBin is not on PATH (-NoModifyPath); add it to your user PATH or use $avExe" }
    elseif (Invoke-Native -Description "add $toolBin to your user PATH" -Argv @($uv, 'tool', 'update-shell') -Soft) {
        Write-Info "open a new terminal for the 'abstractvoice' command to be on PATH"
    } else { Write-Info "add $toolBin to your user PATH yourself (Settings > System > About > Advanced system settings > Environment Variables)" }

    Write-Host ''
    Write-Host $(if ($script:DryRun) { 'Plan (-Print: nothing was changed)' } else { 'AbstractVoice is installed' }) -ForegroundColor White
    $ver = if ($script:DryRun -or -not $after) { $target } else { $after }
    Write-Host "  version:   $ver$(if (-not $script:DryRun -and $before -and $before -ne $after) { " (was $before)" })"
    Write-Host "  profile:   $profileName ($(Get-Spec $fullExtras))"
    Write-Host "  commands:  $avExe, $(Join-Path $toolBin 'abstractvoice-prefetch.exe')"
    Write-Host "  CUDA step: $cudaResult"
    if ($localWhy) { Write-Host "  local engines: skipped ($localWhy)" }
    if (-not $script:DryRun) { Write-Host "  log:       $($script:LogFile)" }
    Write-Host ''
    if ($extras) {
        Write-Host 'Try it (the first command downloads the Supertonic voice once, about 380 MB):'
        Write-Host '  abstractvoice-prefetch --supertonic'
        Write-Host '  abstractvoice --tts-engine supertonic --prompt "Hello from AbstractVoice." --output hello.wav'
    } else { Write-Host 'Try it: abstractvoice --help' }
    Write-Host "Docs: $AvDocs"
}

# The caller's $ErrorActionPreference is not ours (`irm | iex` shares the caller's scope; under
# 'Stop', PowerShell 5.1 turns any native stderr line into a terminating error), so Main runs under
# 'Continue', as with -File, and the caller's value comes back after.
$AvCallerErrorAction = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
try { Main } catch {
    Write-Host ''
    Write-Host "ERROR: $("$_" -replace '^AVBOOT: ', '')" -ForegroundColor Red
    if ("$_" -notlike 'AVBOOT:*') { Write-Host $_.ScriptStackTrace -ForegroundColor DarkGray }
    # `exit` would close the window of someone who pasted `irm | iex`; only exit when run as a file.
    if ($script:RanAsFile) { exit 1 }
} finally {
    $ErrorActionPreference = $AvCallerErrorAction
}
