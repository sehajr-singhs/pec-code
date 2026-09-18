@echo off
cd /d %~dp0
:waitloop
rem wait until every python worker (main farm + e7 side farm) has exited
powershell -NoProfile -Command "if (-not (Get-Process python -ErrorAction SilentlyContinue)) { exit 0 } else { exit 1 }" >nul 2>&1
if errorlevel 1 (
  timeout /t 300 /nobreak >nul
  goto waitloop
)
rem hold_v2 follow-up farm: 10 seeds of the approach-curriculum hold task
python -u seed_farm.py e5v2 --seeds 0-9 --workers 6 --threads 3 --profile farm > farm_log_holdv2.txt 2>&1
