@echo off
rem Laya x Crafter GUI  ->  http://127.0.0.1:8765
set PYTHONUTF8=1
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
cd /d "%~dp0"
start "" http://127.0.0.1:8765
"C:\Users\seann\miniconda3\envs\laya\python.exe" gui\server.py 8765
