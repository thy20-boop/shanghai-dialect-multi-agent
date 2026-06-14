param(
    [string]$OutputDir = "external\ffmpeg-shared",
    [string]$Url = "https://github.com/BtbN/FFmpeg-Builds/releases/latest/download/ffmpeg-master-latest-win64-gpl-shared.zip"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

if (Test-Path (Join-Path $OutputDir "bin\ffmpeg.exe")) {
    Write-Host "FFmpeg shared build already exists: $OutputDir"
    return
}

New-Item -ItemType Directory -Force -Path "work" | Out-Null
$ZipPath = "work\ffmpeg-shared.zip"
$ExtractDir = "work\ffmpeg-shared-extract"

if (Test-Path $ZipPath) {
    Remove-Item -LiteralPath $ZipPath -Force
}
if (Test-Path $ExtractDir) {
    Remove-Item -LiteralPath $ExtractDir -Recurse -Force
}

Write-Host "Downloading FFmpeg shared build..."
Invoke-WebRequest -Uri $Url -OutFile $ZipPath

Write-Host "Extracting FFmpeg shared build..."
Expand-Archive -Path $ZipPath -DestinationPath $ExtractDir -Force
$FfmpegExe = Get-ChildItem -Path $ExtractDir -Recurse -Filter ffmpeg.exe | Select-Object -First 1
if (-not $FfmpegExe) {
    throw "ffmpeg.exe not found in extracted archive."
}
$RootDir = Split-Path (Split-Path $FfmpegExe.FullName -Parent) -Parent

if (Test-Path $OutputDir) {
    Remove-Item -LiteralPath $OutputDir -Recurse -Force
}
New-Item -ItemType Directory -Force -Path (Split-Path $OutputDir -Parent) | Out-Null
Move-Item -LiteralPath $RootDir -Destination $OutputDir

Remove-Item -LiteralPath $ZipPath -Force
Remove-Item -LiteralPath $ExtractDir -Recurse -Force

Write-Host "FFmpeg shared build installed: $OutputDir"
Write-Host "Bin path: $(Resolve-Path (Join-Path $OutputDir 'bin'))"
