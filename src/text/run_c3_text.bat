@echo off
setlocal
rem ======================================================================
rem  C3 (EuqVid-Text): rebuild the text tables, features and charts.
rem  First run: a few minutes, because step 3 parses the lines with spaCy.
rem  Later runs: about a minute; step 3 is skipped when its inputs have not
rem  changed. Inputs are only read; every output is overwritten.
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
%PY% -c "import pandas, matplotlib, spacy" >nul 2>nul || goto missing

echo [1/6] Utterance tables for MELD and AMI ...
%PY% -u build_utterances.py || goto failed
echo.
echo [2/6] Lexical features and confidence intervals ...
%PY% -u lexical_features.py || goto failed
echo.
echo [3/6] Parsing the lines with spaCy (a few minutes the first time) ...
%PY% -u parse_frames.py || goto failed
echo.
echo [4/6] Agency and power features ...
%PY% -u agency_power_features.py || goto failed
echo.
echo [5/6] Charts ...
%PY% -u plot_lexical.py || goto failed
echo.
echo [6/6] Hand-worked tests ...
%PY% -u test_text_features.py || goto failed
echo.
echo All done. Results are in equvid\components\C3_Text\results
pause
exit /b 0

:missing
echo pandas, matplotlib or spaCy is missing for this Python.
echo Run equvid\components\C3_Text\setup\setup_c3.bat once, then run this again.
pause
exit /b 1

:failed
echo.
echo A step stopped with an error - copy the lines above and send them to Claude.
pause
exit /b 1
