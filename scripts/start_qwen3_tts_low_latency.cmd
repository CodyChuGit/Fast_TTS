@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_qwen3_tts_low_latency.ps1" -OpenBrowser
if errorlevel 1 pause
