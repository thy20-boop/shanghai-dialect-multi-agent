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

& $python scripts/check_environment.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python -m ganagent.cli demo
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python -m ganagent.cli evaluate --predictions data/examples/eval_pairs.jsonl
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python scripts/run_pipeline.py --mode mock
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python scripts/run_tests.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python -m compileall src scripts tests app
exit $LASTEXITCODE
