param(
  [Parameter(Mandatory=$true)]
  [string]$Audio,
  [ValidateSet("dolphin_multiagent", "hybrid", "dolphin", "whisper", "funasr", "mock")]
  [string]$Backend = "dolphin_multiagent",
  [string]$Model = "",
  [double]$ChunkSeconds = 15.0,
  [double]$MaxSpeechRegionSeconds = 8.0,
  [switch]$NoVad,
  [string[]]$CustomRepair = @(),
  [string]$ActiveLearningLog = "data/active_learning_queue.jsonl",
  [switch]$NoSaveActiveLearning,
  [string]$TtsOutput = "",
  [ValidateSet("mandarin", "wuu")]
  [string]$TtsTarget = "mandarin",
  [string]$TtsVoice = "zh-CN-XiaoxiaoNeural",
  [string]$TtsRate = "+0%",
  [string]$CodexTaskOutput = "",
  [switch]$Json,
  [switch]$Online
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$cmd = @(
  "translate",
  "--audio", $Audio,
  "--backend", $Backend,
  "--chunk-seconds", "$ChunkSeconds",
  "--max-speech-region-seconds", "$MaxSpeechRegionSeconds"
)
if ($Model) {
  $cmd += @("--model", $Model)
}
if ($NoVad) {
  $cmd += "--no-vad"
}
if ($ActiveLearningLog) {
  $cmd += @("--active-learning-log", $ActiveLearningLog)
}
if ($NoSaveActiveLearning) {
  $cmd += "--no-save-active-learning"
}
if ($TtsOutput) {
  $cmd += @(
    "--tts-output", $TtsOutput,
    "--tts-target", $TtsTarget,
    "--tts-voice", $TtsVoice,
    "--tts-rate", $TtsRate
  )
}
if ($CodexTaskOutput) {
  $cmd += @("--codex-task-output", $CodexTaskOutput)
}
foreach ($repair in $CustomRepair) {
  if ($repair) {
    $cmd += @("--custom-repair", $repair)
  }
}
if ($Json) {
  $cmd += "--json"
}
if (-not $Online) {
  $cmd += "--local-files-only"
}

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

& $python -m ganagent.cli @cmd
if ($LASTEXITCODE -ne 0) {
  throw "Shanghai dialect translation failed with exit code $LASTEXITCODE."
}
