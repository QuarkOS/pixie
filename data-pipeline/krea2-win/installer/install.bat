@echo off
rem One-click installer: double-click this file. Everything goes into this folder (no admin, no global installs).
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
echo.
pause
