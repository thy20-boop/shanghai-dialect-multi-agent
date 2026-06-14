$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONPATH = "src"

$python = ".venv\Scripts\python.exe"
if (Test-Path $python) {
  try {
    & $python -c "import sys" *> $null
    if ($LASTEXITCODE -ne 0) {
      $python = "python"
    }
  }
  catch {
    $python = "python"
  }
}
if (-not (Test-Path $python) -and $python -ne "python") {
  $python = "python"
}

& $python -m ganagent.cli transcribe `
  --audio data\shanghai_audio\shanghai_000002.wav `
  --backend whisper `
  --model TingChen-ppmc/whisper-small-Shanghai `
  --local-files-only `
  --markdown outputs\real_asr_smoke_report.md
exit $LASTEXITCODE
