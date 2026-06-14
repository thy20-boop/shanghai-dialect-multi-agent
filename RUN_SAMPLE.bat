@echo off
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONPATH=%CD%\src
if not exist ".venv\Scripts\python.exe" (
  echo First run: installing Python dependencies. This can take several minutes.
  powershell -ExecutionPolicy Bypass -File scripts\setup_colleague.ps1
  if errorlevel 1 pause & exit /b 1
)
.venv\Scripts\python.exe -c "import torch, transformers, soundfile, peft, imageio_ffmpeg" >nul 2>nul
if errorlevel 1 (
  echo Installing missing Python dependencies. This can take several minutes.
  powershell -ExecutionPolicy Bypass -File scripts\setup_colleague.ps1
  if errorlevel 1 pause & exit /b 1
)
.venv\Scripts\python.exe -m ganagent.cli translate --audio data\shanghai_audio\shanghai_000002.wav --backend whisper --model outputs\models\whisper-small-shanghai-lora-full --json
set EXIT_CODE=%ERRORLEVEL%
echo.
pause
exit /b %EXIT_CODE%
