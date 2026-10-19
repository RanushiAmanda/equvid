@echo off
setlocal
title EuqVid dataset download - AMI then AVA
rem Double-click to start. Safe to stop with Ctrl+C and double-click again:
rem both scripts resume where they stopped and skip files that are complete.
cd /d "%~dp0"

rem ---- find Python -------------------------------------------------------
set "PY="
where python >nul 2>nul && set "PY=python"
if not defined PY (
  where py >nul 2>nul && set "PY=py -3"
)
if not defined PY (
  echo Python was not found. Install Python 3 from python.org and tick "Add python.exe to PATH".
  pause
  exit /b 1
)

rem ---- free space on this drive ------------------------------------------
set "DRV=%~d0"
set "DRV=%DRV::=%"
set "FREE="
for /f %%a in ('powershell -NoProfile -Command "[math]::Floor((Get-PSDrive -Name %DRV%).Free/1GB)"') do set "FREE=%%a"
if not defined FREE (
  echo Could not read free space on %~d0 - continuing.
) else (
  echo Free space on %~d0  %FREE% GB   ^(about 75 GB is needed for everything^)
  if %FREE% LSS 75 (
    echo.
    echo WARNING: under 75 GB free. AMI will fit; all of AVA may not.
    choice /M "Continue anyway"
    if errorlevel 2 exit /b 1
  )
)

set "PYTHONUTF8=1"

echo.
echo ===== Step 1 of 2: AMI room-view video, mixed-gender series (about 5 GB) =====
%PY% -u download_ami.py

echo.
echo ===== Step 2 of 2: AVA-ActiveSpeaker videos (about 40-67 GB) =====
%PY% -u download_ava.py

echo.
echo ===== Final status =====
%PY% -u download_ami.py --list
echo.
%PY% -u download_ava.py --list
echo.
echo Finished. If anything says FAILED or "partial", double-click this file again to retry.
pause
