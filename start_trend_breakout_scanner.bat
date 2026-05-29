@echo off
chcp 65001 >nul
setlocal

title BTC 4H Trend Breakout Scanner
cd /d "%~dp0"

if not defined PROXY_URL set "PROXY_URL=http://127.0.0.1:7897"
if not defined WATCH_SECONDS set "WATCH_SECONDS=60"
if not defined INST_ID set "INST_ID=BTC-USDT-SWAP"
if not defined INITIAL_EQUITY set "INITIAL_EQUITY=1000"
if not defined LEVERAGE set "LEVERAGE=8"
if not defined MARGIN_PCT set "MARGIN_PCT=0.15"

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

echo Starting BTC-USDT-SWAP 4H trend breakout scanner...
echo Instrument: %INST_ID%
echo Refresh interval: %WATCH_SECONDS% seconds
echo Account equity: %INITIAL_EQUITY% USDT
echo Leverage: %LEVERAGE%x
echo Margin pct: %MARGIN_PCT%
echo Proxy fallback: %PROXY_URL%
echo Press Ctrl+C to stop.
echo.

"%PYTHON_CMD%" trend_breakout_opportunity_scanner.py --inst-id "%INST_ID%" --watch %WATCH_SECONDS% --initial-equity %INITIAL_EQUITY% --leverage %LEVERAGE% --margin-pct %MARGIN_PCT% --proxy-mode fallback --proxy-url "%PROXY_URL%"

echo.
echo Program exited.
if not defined NO_PAUSE pause
