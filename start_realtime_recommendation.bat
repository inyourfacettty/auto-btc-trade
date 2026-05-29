@echo off
chcp 65001 >nul
setlocal

title BTC Realtime Recommendation
cd /d "%~dp0"

if not defined PROXY_URL set "PROXY_URL=http://127.0.0.1:7897"
if not defined WATCH_SECONDS set "WATCH_SECONDS=30"
if not defined INST_ID set "INST_ID=BTC-USDT-SWAP"

where python >nul 2>nul
if errorlevel 1 (
    where py >nul 2>nul
    if errorlevel 1 (
        echo Python was not found. Please install Python or add it to PATH.
        echo.
        if not defined NO_PAUSE pause
        exit /b 1
    )
    set "PYTHON_CMD=py"
) else (
    set "PYTHON_CMD=python"
)

echo Starting BTC-USDT realtime recommendation...
echo Instrument: %INST_ID%
echo Refresh interval: %WATCH_SECONDS% seconds
echo Proxy fallback: %PROXY_URL%
echo Press Ctrl+C to stop.
echo.

"%PYTHON_CMD%" realtime_recommendation.py --inst-id "%INST_ID%" --watch %WATCH_SECONDS% --proxy-mode fallback --proxy-url "%PROXY_URL%"

echo.
echo Program exited.
if not defined NO_PAUSE pause
