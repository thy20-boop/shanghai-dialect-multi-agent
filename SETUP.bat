@echo off
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
powershell -ExecutionPolicy Bypass -File scripts\setup_colleague.ps1
if errorlevel 1 pause
