@echo off
setlocal
rem ======================================================================
rem  C3 (EuqVid-Text) setup - installs what C3 needs on this PC and then
rem  checks that it all works. Safe to run again at any time.
rem
rem  Installs: spaCy + English model (en_core_web_sm), VADER, ffmpeg
rem  (imageio-ffmpeg). Total download roughly 100 MB.
rem  The agency/power lexicon is not installed here - it is a file, already
rem  saved in "Research Project\Lexicons\sap2017_agency_power".
rem ======================================================================
cd /d "%~dp0"
set PYTHONUTF8=1

rem --- find Python: same rule as equvid\data\download_all.bat -----------
set "PY="
where python >nul 2>nul && set "PY=python"
if not defined PY (where py >nul 2>nul && set "PY=py -3")
if not defined PY (
  echo Python was not found. Install Python 3.12 from python.org, tick
  echo "Add python.exe to PATH" during installation, then run this again.
  pause
  exit /b 1
)
echo Using this Python - C3 code must be run with the same one:
%PY% -c "import sys; print('  ', sys.executable); print('   Python', sys.version.split()[0])"
echo.

echo [1/3] Installing spaCy, VADER and ffmpeg ...
%PY% -m pip install -r requirements-c3.txt
if errorlevel 1 goto failed

echo.
echo [2/3] Downloading the spaCy English model en_core_web_sm ...
%PY% -m spacy download en_core_web_sm
if errorlevel 1 goto failed

echo.
echo [3/3] Checking everything works (about a minute) ...
echo.
%PY% -u check_c3_setup.py
echo.
echo These results are also saved in c3_setup_report.txt in this folder.
pause
exit /b 0

:failed
echo.
echo Installation stopped with an error - read the lines above.
echo If they say "No matching distribution" or "Microsoft Visual C++ is required",
echo this Python is probably newer than spaCy supports. Install Python 3.12 from
echo python.org and run, in this folder:
echo     py -3.12 -m pip install -r requirements-c3.txt
echo     py -3.12 -m spacy download en_core_web_sm
echo     py -3.12 check_c3_setup.py
pause
exit /b 1
