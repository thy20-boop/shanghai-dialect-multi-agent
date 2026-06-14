@echo off
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONPATH=%CD%\src
set STREAMLIT_SERVER_SHOW_EMAIL_PROMPT=false
set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
if not defined SHANGHAI_WU_RUNTIME (
  if exist "D:\wswu_runtime" (
    set "SHANGHAI_WU_RUNTIME=D:\wswu_runtime"
  ) else (
    set "SHANGHAI_WU_RUNTIME=%LOCALAPPDATA%\ShanghaiDialectAgent\wswu_runtime"
  )
)
if not exist ".venv\Scripts\python.exe" (
  echo First run: installing Python dependencies. This can take several minutes.
  powershell -ExecutionPolicy Bypass -File scripts\setup_colleague.ps1
  if errorlevel 1 pause & exit /b 1
)
.venv\Scripts\python.exe -c "import streamlit, torch, torchaudio, transformers, soundfile, peft, imageio_ffmpeg, dolphin, funasr, langid, whisper" >nul 2>nul
if errorlevel 1 (
  echo Installing missing Python dependencies. This can take several minutes.
  powershell -ExecutionPolicy Bypass -File scripts\setup_colleague.ps1
  if errorlevel 1 pause & exit /b 1
)
if not exist "%SHANGHAI_WU_RUNTIME%\models\Whisper-Medium-Wu\whisper\whisper-medium.pt" (
  echo First run: downloading the official Whisper-Medium-Wu expert model.
  powershell -ExecutionPolicy Bypass -File scripts\setup_whisper_medium_wu.ps1
  if errorlevel 1 pause & exit /b 1
)
if not exist "%SHANGHAI_WU_RUNTIME%\models\CosyVoice2-Wu-SFT-runtime\cosyvoice2.yaml" (
  echo First run: downloading the official CosyVoice2-Wu-SFT generation expert.
  powershell -ExecutionPolicy Bypass -File scripts\setup_wenet_wu_sft.ps1 -RuntimeRoot "%SHANGHAI_WU_RUNTIME%"
  if errorlevel 1 pause & exit /b 1
)
echo Starting Shanghai Dialect ASR at http://localhost:8501
powershell -NoProfile -Command "if (-not (Test-NetConnection 127.0.0.1 -Port 9881 -WarningAction SilentlyContinue).TcpTestSucceeded) { powershell -ExecutionPolicy Bypass -File scripts\start_wenet_wu_expert.ps1 -Detached }"
.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py --server.port 8501 --server.showEmailPrompt false --browser.gatherUsageStats false
if errorlevel 1 pause
