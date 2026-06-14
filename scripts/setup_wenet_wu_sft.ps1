param(
    [string]$RuntimeRoot = $env:SHANGHAI_WU_RUNTIME,
    [switch]$SkipDependencies
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$legacyRuntime = "D:\wswu_runtime"
if (-not $RuntimeRoot) {
    $RuntimeRoot = if (Test-Path $legacyRuntime) { $legacyRuntime } else { Join-Path $env:LOCALAPPDATA "ShanghaiDialectAgent\wswu_runtime" }
}
$RuntimeRoot = [System.IO.Path]::GetFullPath($RuntimeRoot)
$env:SHANGHAI_WU_RUNTIME = $RuntimeRoot
$env:MODELSCOPE_CACHE = Join-Path $RuntimeRoot "modelscope_cache"
$CosyVoiceDir = Join-Path $RuntimeRoot "CosyVoice"
$RuntimePython = Join-Path $RuntimeRoot ".venv\Scripts\python.exe"
New-Item -ItemType Directory -Force -Path $RuntimeRoot, $env:MODELSCOPE_CACHE | Out-Null

if (-not (Test-Path (Join-Path $CosyVoiceDir "cosyvoice\cli\cosyvoice.py"))) {
    Write-Host "Cloning official CosyVoice runtime..."
    & git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git $CosyVoiceDir
    if ($LASTEXITCODE -ne 0) { throw "CosyVoice clone failed." }
}
& git -C $CosyVoiceDir checkout c93d3dda01ae69fde8a4b2372f0e4260135599c3
& git -C $CosyVoiceDir submodule update --init --recursive
if ($LASTEXITCODE -ne 0) { throw "CosyVoice source checkout failed." }

if (-not (Test-Path $RuntimePython)) {
    $conda = Get-Command conda -ErrorAction SilentlyContinue
    if ($conda) {
        & conda create -p (Join-Path $RuntimeRoot ".venv") -y python=3.10
    }
    else {
        & py -3.10 -m venv (Join-Path $RuntimeRoot ".venv")
    }
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $RuntimePython)) {
        throw "Python 3.10 environment creation failed. Install Miniconda or Python 3.10 and retry."
    }
}

if (-not $SkipDependencies) {
    Write-Host "Installing official CosyVoice dependencies..."
    & $RuntimePython -m ensurepip --upgrade
    & $RuntimePython -m pip install --upgrade pip wheel setuptools
    & $RuntimePython -m pip install -r (Join-Path $CosyVoiceDir "requirements.txt")
    & $RuntimePython -m pip install huggingface_hub fastapi uvicorn
    if ($LASTEXITCODE -ne 0) { throw "CosyVoice dependency installation failed." }
}

Write-Host "Downloading official WenetSpeech-Wu SFT model. This requires several GB..."
& $RuntimePython (Join-Path $ProjectRoot "scripts\download_wenet_wu_sft.py") --runtime-root $RuntimeRoot
if ($LASTEXITCODE -ne 0) { throw "WenetSpeech-Wu SFT download failed." }

$PromptSource = Join-Path $ProjectRoot "assets\reference\official_shanghai_prompt.wav"
Copy-Item -LiteralPath $PromptSource -Destination (Join-Path $RuntimeRoot "prompt_shanghai.wav") -Force

Write-Host "WenetSpeech-Wu SFT runtime ready: $RuntimeRoot"
Write-Host "Start it with:"
Write-Host "  powershell -ExecutionPolicy Bypass -File scripts\start_wenet_wu_expert.ps1 -RuntimeRoot `"$RuntimeRoot`" -Detached"
