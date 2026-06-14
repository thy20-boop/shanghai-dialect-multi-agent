param(
  [string]$Version = (Get-Date -Format "yyyyMMdd-HHmm")
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$releaseRoot = Join-Path $root "outputs\releases"
$packageName = "shanghai-dialect-agent-portable-$Version"
$stageParent = Join-Path ([System.IO.Path]::GetTempPath()) "shanghai-agent-build"
$stage = Join-Path $stageParent $packageName
$zip = Join-Path $releaseRoot "$packageName.zip"
$runtimeZip = Join-Path $releaseRoot "python-3.12.10-embed-amd64.zip"
$runtimeUrl = "https://www.python.org/ftp/python/3.12.10/python-3.12.10-embed-amd64.zip"
$modelCache = Join-Path $env:USERPROFILE ".cache\huggingface\hub\models--TingChen-ppmc--whisper-small-Shanghai"

New-Item -ItemType Directory -Force $releaseRoot | Out-Null
New-Item -ItemType Directory -Force $stageParent | Out-Null

$resolvedStageParent = [System.IO.Path]::GetFullPath($stageParent)
$resolvedStage = [System.IO.Path]::GetFullPath($stage)
if (-not $resolvedStage.StartsWith($resolvedStageParent)) {
  throw "Refusing to clean unexpected path: $resolvedStage"
}

if (Test-Path $stage) {
  [System.IO.Directory]::Delete("\\?\$resolvedStage", $true)
}
if (Test-Path $zip) {
  Remove-Item -LiteralPath $zip -Force
}

if (-not (Test-Path $runtimeZip)) {
  Invoke-WebRequest -Uri $runtimeUrl -OutFile $runtimeZip
}

New-Item -ItemType Directory -Force $stage | Out-Null
New-Item -ItemType Directory -Force (Join-Path $stage "runtime") | Out-Null
Expand-Archive -LiteralPath $runtimeZip -DestinationPath (Join-Path $stage "runtime")

$runtimeSitePackages = Join-Path $stage "runtime\Lib\site-packages"
New-Item -ItemType Directory -Force $runtimeSitePackages | Out-Null
$sourceSitePackages = (Resolve-Path ".venv\Lib\site-packages").Path
$streamlitAgentAssets = Join-Path $sourceSitePackages "streamlit\.agents"
& robocopy.exe $sourceSitePackages $runtimeSitePackages /E /XD $streamlitAgentAssets /NFL /NDL /NJH /NJS /NP
if ($LASTEXITCODE -gt 7) {
  throw "Copying portable site-packages failed with robocopy exit code $LASTEXITCODE."
}
Get-ChildItem $runtimeSitePackages -Filter "__editable__*.pth" -ErrorAction SilentlyContinue |
  Remove-Item -Force

$pthPath = Join-Path $stage "runtime\python312._pth"
@(
  "python312.zip"
  "."
  "Lib"
  "Lib\site-packages"
  "..\src"
  "import site"
) | Set-Content -LiteralPath $pthPath -Encoding ASCII

foreach ($dir in @("app", "configs", "docs", "src")) {
  Copy-Item -LiteralPath $dir -Destination $stage -Recurse
}

New-Item -ItemType Directory -Force (Join-Path $stage "data") | Out-Null
Copy-Item -LiteralPath "data\examples" -Destination (Join-Path $stage "data") -Recurse
Copy-Item -LiteralPath "data\shanghai_audio" -Destination (Join-Path $stage "data") -Recurse
Copy-Item -LiteralPath "data\shanghai_manifest.jsonl" -Destination (Join-Path $stage "data")
if (Test-Path -LiteralPath "data\splits") {
  Copy-Item -LiteralPath "data\splits" -Destination (Join-Path $stage "data") -Recurse
}

New-Item -ItemType Directory -Force (Join-Path $stage "outputs") | Out-Null
foreach ($file in @(
  "outputs\shanghai_10_predictions.jsonl",
  "outputs\shanghai_real_asr_report.md"
)) {
  if (Test-Path $file) {
    Copy-Item -LiteralPath $file -Destination (Join-Path $stage "outputs")
  }
}
Get-ChildItem -LiteralPath "outputs" -Filter "dev_eval_92*" -File -ErrorAction SilentlyContinue |
  ForEach-Object { Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $stage "outputs") }

$trainedModel = "outputs\models\whisper-small-shanghai-lora-full"
$portableTrainedModel = Join-Path $stage "outputs\models\whisper-small-shanghai-lora-full"
if (Test-Path -LiteralPath $trainedModel) {
  New-Item -ItemType Directory -Force (Join-Path $stage "outputs\models") | Out-Null
  Copy-Item -LiteralPath $trainedModel -Destination (Join-Path $stage "outputs\models") -Recurse
}

if (-not (Test-Path (Join-Path $modelCache "refs\main"))) {
  throw "Shanghai model cache not found: $modelCache"
}
$revision = (Get-Content -LiteralPath (Join-Path $modelCache "refs\main") -Raw).Trim()
$modelSnapshot = Join-Path $modelCache "snapshots\$revision"
$portableBaseModel = Join-Path $stage "models\whisper-small-Shanghai"
New-Item -ItemType Directory -Force $portableBaseModel | Out-Null
Copy-Item -Path (Join-Path $modelSnapshot "*") -Destination $portableBaseModel -Recurse -Force

Copy-Item -LiteralPath "portable\START_UI.bat" -Destination $stage
Copy-Item -LiteralPath "portable\TRANSCRIBE_AUDIO.bat" -Destination $stage
Copy-Item -LiteralPath "portable\RUN_SAMPLE.bat" -Destination $stage
Copy-Item -LiteralPath "portable\PORTABLE_README.txt" -Destination $stage
Copy-Item -LiteralPath "README.md" -Destination $stage
Copy-Item -LiteralPath "requirements-experimental-asr.txt" -Destination $stage

Get-ChildItem $stage -Recurse -Directory -Force |
  Where-Object { $_.Name -in @("__pycache__", ".pytest_cache") -or $_.Name.EndsWith(".egg-info") } |
  ForEach-Object { [System.IO.Directory]::Delete("\\?\$($_.FullName)", $true) }

$env:PYTHONUTF8 = "1"
$env:PYTHONPATH = Join-Path $stage "src"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:SHANGHAI_ASR_MODEL = $portableTrainedModel
$portablePython = Join-Path $stage "runtime\python.exe"

& $portablePython -c "import torch, transformers, streamlit; print(torch.__version__, transformers.__version__, streamlit.__version__)"
if ($LASTEXITCODE -ne 0) {
  throw "Portable runtime import check failed."
}

& $portablePython -m ganagent.cli translate `
  --audio (Join-Path $stage "data\shanghai_audio\shanghai_000002.wav") `
  --backend whisper `
  --model $portableTrainedModel `
  --local-files-only `
  --json
if ($LASTEXITCODE -ne 0) {
  throw "Portable Shanghai ASR sample check failed."
}

tar.exe -a -c -f $zip -C $stageParent $packageName
if ($LASTEXITCODE -ne 0) {
  throw "Creating portable zip failed."
}

$hash = (Get-FileHash -Algorithm SHA256 $zip).Hash
"SHA256  $hash  $packageName.zip" | Set-Content -Encoding ASCII (Join-Path $releaseRoot "$packageName.sha256.txt")

$folderBytes = (Get-ChildItem -Recurse -File $stage | Measure-Object Length -Sum).Sum
$zipBytes = (Get-Item $zip).Length
Write-Host "Portable folder: $stage"
Write-Host "Portable zip: $zip"
Write-Host "Folder size MB: $([math]::Round($folderBytes / 1MB, 2))"
Write-Host "Zip size MB: $([math]::Round($zipBytes / 1MB, 2))"
Write-Host "SHA256: $hash"
