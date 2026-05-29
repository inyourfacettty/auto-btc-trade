@echo off
chcp 65001 >nul
setlocal

title BTC Local Auto Trader UI
cd /d "%~dp0"

if not defined PROXY_URL set "PROXY_URL=http://127.0.0.1:7897"
if not defined INST_ID set "INST_ID=BTC-USDT-SWAP"
if not defined INITIAL_EQUITY set "INITIAL_EQUITY=1000"
if not defined LEVERAGE set "LEVERAGE=8"
if not defined MARGIN_PCT set "MARGIN_PCT=0.10"
if not defined PORT set "PORT=8765"
if not defined ENV_PATH set "ENV_PATH=.env"

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

echo 正在启动 BTC 本机自动交易 UI...
echo 地址: http://127.0.0.1:%PORT%
echo 品种: %INST_ID%
echo 本金: %INITIAL_EQUITY% USDT
echo 杠杆: %LEVERAGE%x
echo 保证金比例: %MARGIN_PCT%
echo 真实盘下单: 关闭
echo OKX 配置文件: %ENV_PATH%
echo 代理: %PROXY_URL%
echo 按 Ctrl+C 停止。
echo.

"%PYTHON_CMD%" local_auto_trader_ui.py --inst-id "%INST_ID%" --port %PORT% --initial-equity %INITIAL_EQUITY% --leverage %LEVERAGE% --margin-pct %MARGIN_PCT% --proxy-mode fallback --proxy-url "%PROXY_URL%" --env-path "%ENV_PATH%"

echo.
echo 程序已退出。
if not defined NO_PAUSE pause
