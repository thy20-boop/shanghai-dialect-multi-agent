param(
    [string]$RuntimeRoot = $env:SHANGHAI_WU_RUNTIME,
    [string]$RuntimePython = "",
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 9881,
    [ValidateSet("sft", "cpt", "prosody")]
    [string]$Expert = "sft",
    [switch]$Fp16,
    [switch]$Detached
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot
$legacyRuntime = "D:\wswu_runtime"
if (-not $RuntimeRoot) {
    $RuntimeRoot = if (Test-Path $legacyRuntime) { $legacyRuntime } else { Join-Path $env:LOCALAPPDATA "ShanghaiDialectAgent\wswu_runtime" }
}
if (-not $RuntimePython) {
    $RuntimePython = Join-Path $RuntimeRoot ".venv\Scripts\python.exe"
}
$env:SHANGHAI_WU_RUNTIME = $RuntimeRoot

if (-not (Test-Path -LiteralPath $RuntimePython)) {
    throw "WenetSpeech-Wu runtime not found: $RuntimePython"
}

$ModelDirs = @{
    sft = Join-Path $RuntimeRoot "models\CosyVoice2-Wu-SFT-runtime"
    cpt = Join-Path $RuntimeRoot "models\CosyVoice2-Wu-CPT-runtime"
    prosody = Join-Path $RuntimeRoot "models\CosyVoice2-Wu-instruct-prosody-runtime"
}
$env:WSWU_MODEL_DIR = $ModelDirs[$Expert]
$env:WSWU_COSYVOICE_DIR = Join-Path $RuntimeRoot "CosyVoice"
$env:WSWU_PROMPT_AUDIO = Join-Path $ProjectRoot "assets\reference\official_shanghai_prompt.wav"
$env:WSWU_EXPERT_MODE = if ($Expert -eq "prosody") { "instruct_prosody" } else { "zero_shot" }
$env:WSWU_FP16 = if ($Fp16) { "1" } else { "0" }
$env:MODELSCOPE_CACHE = Join-Path $RuntimeRoot "modelscope_cache"
if (-not (Test-Path -LiteralPath $env:WSWU_MODEL_DIR)) {
    throw "Wu expert model not found: $($env:WSWU_MODEL_DIR). Run scripts\setup_wenet_wu_sft.ps1 first."
}

$Args = @(
    "-m", "uvicorn",
    "scripts.wenet_wu_server:app",
    "--host", $HostAddress,
    "--port", "$Port"
)

if ($Detached) {
    $Process = Start-Process -FilePath $RuntimePython -ArgumentList $Args -WorkingDirectory $ProjectRoot -WindowStyle Hidden -PassThru
    Write-Host "Started WenetSpeech-Wu expert. PID=$($Process.Id)"
    Write-Host "Expert: $Expert"
    Write-Host "URL: http://$HostAddress`:$Port/tts"
} else {
    & $RuntimePython @Args
    if ($LASTEXITCODE -ne 0) {
        throw "WenetSpeech-Wu expert exited with code $LASTEXITCODE."
    }
}
