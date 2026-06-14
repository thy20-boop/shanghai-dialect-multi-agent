param(
  [ValidateSet("data", "asr", "finetune", "ui", "online", "ocr", "tts")]
  [string]$Group = "data",
  [ValidateSet("auto", "cuda", "cpu")]
  [string]$TorchBackend = "auto"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$venvPython = ".venv\Scripts\python.exe"
$venvHealthy = Test-Path $venvPython
if ($venvHealthy) {
  try {
    & $venvPython -c "import sys" *> $null
    $venvHealthy = $LASTEXITCODE -eq 0
  }
  catch {
    $venvHealthy = $false
  }
}

if (-not $venvHealthy) {
  if (Test-Path ".venv") {
    $resolvedVenv = [System.IO.Path]::GetFullPath((Join-Path $root ".venv"))
    $resolvedRoot = [System.IO.Path]::GetFullPath($root)
    if (-not $resolvedVenv.StartsWith($resolvedRoot)) {
      throw "Refusing to replace unexpected virtualenv path: $resolvedVenv"
    }
    Remove-Item -LiteralPath $resolvedVenv -Recurse -Force
  }
  python -m venv .venv
}

if ($Group -eq "asr" -or $Group -eq "finetune") {
  $selectedTorchBackend = $TorchBackend
  if ($selectedTorchBackend -eq "auto") {
    $hasNvidia = $false
    try {
      & nvidia-smi *> $null
      $hasNvidia = $LASTEXITCODE -eq 0
    }
    catch {
      $hasNvidia = $false
    }
    $selectedTorchBackend = if ($hasNvidia) { "cuda" } else { "cpu" }
  }

  if ($selectedTorchBackend -eq "cuda") {
    & $venvPython -m pip install "torch==2.6.0+cu126" --index-url https://download.pytorch.org/whl/cu126 --progress-bar off --no-cache-dir
  }
  else {
    & $venvPython -m pip install "torch==2.6.0+cpu" --index-url https://download.pytorch.org/whl/cpu --progress-bar off --no-cache-dir
  }
}

$requirements = "requirements-$Group.txt"
& $venvPython -m pip install -r $requirements --progress-bar off --no-cache-dir
& $venvPython -m pip install -e . --progress-bar off --no-cache-dir
