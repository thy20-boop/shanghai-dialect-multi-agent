param(
  [switch]$WithOnline,
  [switch]$WithOcr,
  [switch]$WithFinetune
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

powershell -ExecutionPolicy Bypass -File scripts\install_deps.ps1 -Group asr
powershell -ExecutionPolicy Bypass -File scripts\install_deps.ps1 -Group ui

if ($WithOnline) {
  powershell -ExecutionPolicy Bypass -File scripts\install_deps.ps1 -Group online
}

if ($WithOcr) {
  powershell -ExecutionPolicy Bypass -File scripts\install_deps.ps1 -Group ocr
}

if ($WithFinetune) {
  powershell -ExecutionPolicy Bypass -File scripts\install_deps.ps1 -Group finetune
}

Write-Host ""
Write-Host "Setup complete."
Write-Host "Run Web UI:"
Write-Host "  .venv\Scripts\python.exe -m streamlit run app\streamlit_app.py --server.port 8501"
Write-Host ""
Write-Host "Run CLI:"
Write-Host "  .\scripts\translate_audio.ps1 -Audio path\to\shanghai.wav -Json -Online"
