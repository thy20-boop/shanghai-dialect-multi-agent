@echo off
cd /d "%~dp0"
if "%~1"=="" goto usage
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
.venv\Scripts\python.exe -m ganagent.cli translate --audio "%~1" --backend whisper --model outputs\models\whisper-small-shanghai-lora-full --json
set EXIT_CODE=%ERRORLEVEL%
echo.
pause
exit /b %EXIT_CODE%

:usage
echo Drag a WAV, FLAC, OGG, M4A, MP3, MP4, or AAC audio file onto TRANSCRIBE_AUDIO.bat.
echo You can also run START_UI.bat for the web interface.
pause
exit /b 1
