@echo off
setlocal
rem ======================================================================
rem  C3 decision D6: does an off-the-shelf sentiment model pass the three
rem  checks fixed in advance? About 5-15 minutes on a laptop CPU.
rem  Needs setup_sentiment.bat to have been run once.
rem  Results: equvid\components\C3_Text\results\c3_sentiment_check.csv
rem ======================================================================
cd /d "%~dp0"
set PYTHONUTF8=1

set "PY="
where python >nul 2>nul && set "PY=python"
if not defined PY (where py >nul 2>nul && set "PY=py -3")
if not defined PY goto missing
%PY% -c "import transformers, torch, pandas" >nul 2>nul || goto missing

echo Running the sentiment check ...
%PY% -u sentiment_check.py || goto failed
echo.
echo Done. Results are in equvid\components\C3_Text\results\c3_sentiment_check.csv
pause
exit /b 0

:missing
echo transformers or PyTorch is not installed for this Python.
echo Run equvid\components\C3_Text\setup\setup_sentiment.bat once, then run this again.
pause
exit /b 1

:failed
echo.
echo The check stopped with an error - copy the lines above and send them to Claude.
pause
exit /b 1
