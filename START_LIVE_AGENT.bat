@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Python virtual environment not found. Running SETUP.bat first...
  call SETUP.bat
)

echo Installing realtime microphone dependencies...
".venv\Scripts\python.exe" -m pip install -r requirements-live.txt

echo Starting Wu speech expert service if available...
powershell -ExecutionPolicy Bypass -File scripts\start_wenet_wu_expert.ps1 -Detached

echo Starting realtime Shanghai Dialect Agent...
".venv\Scripts\python.exe" -m ganagent.live_agent --reply-target wuu --json-log outputs\live_agent\dialogue_log.jsonl

pause
