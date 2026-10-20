@echo off
setlocal
rem ======================================================================
rem  C3 decision D6 - the sentiment check. Installs transformers and PyTorch
rem  (CPU build), then downloads the sentiment model once.
rem  Total download about 1 GB. Safe to run again.
rem ======================================================================
cd /d "%~dp0"
set PYTHONUTF8=1

rem --- find Python: same rule as setup_c3.bat ----------------------------
set "PY="
where python >nul 2>nul && set "PY=python"
if not defined PY (where py >nul 2>nul && set "PY=py -3")
if not defined PY goto nopython

echo [1/2] Installing transformers and PyTorch (CPU build) ...
%PY% -m pip install -r requirements-sentiment.txt || goto failed
echo.
echo [2/2] Downloading the sentiment model (about 500 MB, only the first time) ...
%PY% -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification as M; n='cardiffnlp/twitter-roberta-base-sentiment-latest'; AutoTokenizer.from_pretrained(n); M.from_pretrained(n); print('model ready')" || goto failed
echo.
echo Done. Now double-click equvid\src\text\run_c3_sentiment.bat
pause
exit /b 0

:nopython
echo Python was not found. Run setup_c3.bat first.
pause
exit /b 1

:failed
echo.
echo The installation stopped with an error - copy the lines above and send them to Claude.
pause
exit /b 1
