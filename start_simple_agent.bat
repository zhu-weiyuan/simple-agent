@echo off
setlocal
REM One-click: SimpleAgent (8000). Uses global python (deps installed there).
REM Prereq: local LLM server running, .env configured
cd /d %~dp0
start "SimpleAgent 8000" cmd /k "python app_prod.py"
timeout /t 5 /nobreak >nul
start http://localhost:8000
echo SimpleAgent http://localhost:8000
endlocal
