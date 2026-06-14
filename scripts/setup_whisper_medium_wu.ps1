param(
  [string]$RuntimeRoot = $env:SHANGHAI_WU_RUNTIME
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$legacyRuntime = "D:\wswu_runtime"
if (-not $RuntimeRoot) {
  $RuntimeRoot = if (Test-Path $legacyRuntime) { $legacyRuntime } else { Join-Path $env:LOCALAPPDATA "ShanghaiDialectAgent\wswu_runtime" }
}
$env:SHANGHAI_WU_RUNTIME = $RuntimeRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
  throw "Project virtual environment not found: $python"
}

$wenetDir = Join-Path $RuntimeRoot "wenet-main"
$modelRoot = Join-Path $RuntimeRoot "models\Whisper-Medium-Wu"
$modelDir = Join-Path $modelRoot "whisper"
New-Item -ItemType Directory -Force -Path $RuntimeRoot, $modelRoot | Out-Null

if (-not (Test-Path (Join-Path $wenetDir "wenet\bin\recognize.py"))) {
  $zip = Join-Path $RuntimeRoot "wenet-main.zip"
  Write-Host "Downloading official WeNet runtime..."
  Invoke-WebRequest `
    -Uri "https://codeload.github.com/wenet-e2e/wenet/zip/refs/heads/main" `
    -OutFile $zip `
    -UseBasicParsing
  Expand-Archive -Path $zip -DestinationPath $RuntimeRoot -Force
}

Write-Host "Installing minimal WeNet inference dependencies..."
& $python -m pip install "setuptools<81" wheel
& $python -m pip install langid sentencepiece textgrid
& $python -m pip install --no-build-isolation openai-whisper==20231117

Write-Host "Downloading official WenetSpeech-Wu Whisper-Medium checkpoint..."
$yamlUrl = "https://hf-mirror.com/ASLP-lab/WenetSpeech-Wu-Speech-Understanding/resolve/main/whisper/train.yaml"
$modelUrl = "https://hf-mirror.com/ASLP-lab/WenetSpeech-Wu-Speech-Understanding/resolve/main/whisper/whisper-medium.pt?download=true"
New-Item -ItemType Directory -Force -Path $modelDir | Out-Null
Invoke-WebRequest -Uri $yamlUrl -OutFile (Join-Path $modelDir "train.yaml") -UseBasicParsing
& curl.exe -L --retry 20 --retry-delay 5 -C - -o (Join-Path $modelDir "whisper-medium.pt") $modelUrl
if ($LASTEXITCODE -ne 0) {
  throw "Whisper-Medium-Wu checkpoint download failed."
}

if (-not (Test-Path (Join-Path $modelDir "whisper-medium.pt"))) {
  throw "Whisper-Medium-Wu checkpoint download did not complete."
}

Write-Host "Whisper-Medium-Wu ready: $modelDir"
