@echo off
setlocal
rem ======================================================================
rem  C3 (EuqVid-Text): rebuild the text tables, features and charts.
rem  Takes about a minute. Safe to run again at any time: inputs are only
rem  read, and every output is overwritten.
rem    tables  -> Research Project\derived\c3\
rem    results -> equvid\components\C3_Text\results\
rem ======================================================================
cd /d "%~dp0"
set PYTHONUTF8=1

rem --- find Python: same rule as setup_c3.bat ----------------------------
set "PY="
where python >nul 2>nul && set "PY=python"
if not defined PY (where py >nul 2>nul && set "PY=py -3")
if not defined PY (
  echo Python was not found. Run equvid\components\C3_Text\setup\setup_c3.bat first.
  pause
  exit /b 1
)

echo [1/4] Utterance tables for MELD and AMI ...
%PY% -u build_utterances.py || goto failed
echo.
echo [2/4] Features and bootstrap confidence intervals ...
%PY% -u lexical_features.py || goto failed
echo.
echo [3/4] Charts ...
%PY% -u plot_lexical.py || goto failed
echo.
echo [4/4] Hand-worked tests ...
%PY% -u test_text_features.py || goto failed
echo.
echo All done. Results are in equvid\components\C3_Text\results
pause
exit /b 0

:failed
echo.
echo A step stopped with an error - copy the lines above and send them to Claude.
echo If the error says "No module named pandas" or "matplotlib", run
echo equvid\components\C3_Text\setup\setup_c3.bat once, then run this again.
pause
exit /b 1
