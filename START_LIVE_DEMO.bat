@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Python virtual environment not found. Running SETUP.bat first...
  call SETUP.bat
)

echo Running mic-free realtime agent course demo...
".venv\Scripts\python.exe" -m ganagent.live_agent --demo-scenario course

echo.
echo Demo report:
echo outputs\live_agent_demo_course\session_report.md
pause
