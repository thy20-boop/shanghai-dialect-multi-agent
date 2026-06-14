param(
  [string]$Version = (Get-Date -Format "yyyyMMdd-HHmm")
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$releaseRoot = Join-Path $root "outputs\releases"
$packageName = "shanghai-dialect-agent-online-$Version"
$stage = Join-Path $releaseRoot $packageName
$zip = Join-Path $releaseRoot "$packageName.zip"

New-Item -ItemType Directory -Force $releaseRoot | Out-Null

$resolvedReleaseRoot = [System.IO.Path]::GetFullPath($releaseRoot)
$resolvedStage = [System.IO.Path]::GetFullPath($stage)
if (-not $resolvedStage.StartsWith($resolvedReleaseRoot)) {
  throw "Refusing to clean unexpected path: $resolvedStage"
}

if (Test-Path $stage) {
  [System.IO.Directory]::Delete("\\?\$resolvedStage", $true)
}
if (Test-Path $zip) {
  Remove-Item -LiteralPath $zip -Force
}

New-Item -ItemType Directory -Force $stage | Out-Null

$topLevelFiles = @(
  "README.md",
  "pyproject.toml",
  "requirements-asr.txt",
  "requirements-data.txt",
  "requirements-experimental-asr.txt",
  "requirements-finetune.txt",
  "requirements-ocr.txt",
  "requirements-online.txt",
  "requirements-ui.txt",
  "ONLINE_README.txt",
  "SETUP.bat",
  "START_UI.bat",
  "RUN_SAMPLE.bat",
  "TRANSCRIBE_AUDIO.bat"
)

foreach ($file in $topLevelFiles) {
  Copy-Item -LiteralPath $file -Destination $stage
}

foreach ($dir in @("app", "configs", "docs", "scripts", "src", "tests")) {
  Copy-Item -LiteralPath $dir -Destination $stage -Recurse
}

New-Item -ItemType Directory -Force (Join-Path $stage "data") | Out-Null
Copy-Item -LiteralPath "data\examples" -Destination (Join-Path $stage "data") -Recurse
Copy-Item -LiteralPath "data\shanghai_audio" -Destination (Join-Path $stage "data") -Recurse
Copy-Item -LiteralPath "data\shanghai_manifest.jsonl" -Destination (Join-Path $stage "data")
if (Test-Path -LiteralPath "data\splits") {
  Copy-Item -LiteralPath "data\splits" -Destination (Join-Path $stage "data") -Recurse
}

$trainedModel = "outputs\models\whisper-small-shanghai-lora-full"
if (Test-Path -LiteralPath $trainedModel) {
  New-Item -ItemType Directory -Force (Join-Path $stage "outputs\models") | Out-Null
  Copy-Item -LiteralPath $trainedModel -Destination (Join-Path $stage "outputs\models") -Recurse
}

New-Item -ItemType Directory -Force (Join-Path $stage "outputs") | Out-Null
Get-ChildItem -LiteralPath "outputs" -Filter "dev_eval_92*" -File -ErrorAction SilentlyContinue |
  ForEach-Object { Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $stage "outputs") }

Get-ChildItem $stage -Recurse -Directory -Force |
  Where-Object { $_.Name -in @("__pycache__", ".pytest_cache") -or $_.Name.EndsWith(".egg-info") } |
  ForEach-Object { [System.IO.Directory]::Delete("\\?\$($_.FullName)", $true) }

tar.exe -a -c -f $zip -C $releaseRoot $packageName
if ($LASTEXITCODE -ne 0) {
  throw "Creating online zip failed."
}
tar.exe -tf $zip > $null
if ($LASTEXITCODE -ne 0) {
  throw "Online zip verification failed."
}

$hash = (Get-FileHash -Algorithm SHA256 $zip).Hash
"SHA256  $hash  $packageName.zip" | Set-Content -Encoding ASCII (Join-Path $releaseRoot "$packageName.sha256.txt")

$folderBytes = (Get-ChildItem -Recurse -File $stage | Measure-Object Length -Sum).Sum
$zipBytes = (Get-Item $zip).Length
Write-Host "Online folder: $stage"
Write-Host "Online zip: $zip"
Write-Host "Folder size MB: $([math]::Round($folderBytes / 1MB, 2))"
Write-Host "Zip size MB: $([math]::Round($zipBytes / 1MB, 2))"
Write-Host "SHA256: $hash"
