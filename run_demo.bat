@echo off
REM Double-click this to run the interactive Jugalbandi demo locally.
REM Needs your downloaded trained adapter unzipped into checkpoints\final
REM (see README's "Run the interactive demo locally" section).

cd /d "%~dp0"

set JUGALBANDI_ADAPTER_REPO=%~dp0checkpoints\final
set JUGALBANDI_BASE_MODEL=unsloth/Qwen2.5-0.5B-Instruct
set JUGALBANDI_INFER_ARM=hidden

if not exist "%JUGALBANDI_ADAPTER_REPO%\adapter_config.json" (
    echo.
    echo Could not find a trained adapter at:
    echo   %JUGALBANDI_ADAPTER_REPO%
    echo.
    echo Download the "final" folder from Google Drive ^(Drive ^> jugalbandi ^>
    echo your run name ^> final ^> right-click ^> Download^) and unzip it so its
    echo files land directly in that checkpoints\final folder, then run this
    echo script again.
    echo.
    pause
    exit /b 1
)

if not exist venv\Scripts\python.exe (
    echo.
    echo No venv found. Run this first in a terminal, in this folder:
    echo   python -m venv venv
    echo   venv\Scripts\activate
    echo   pip install -r requirements.txt -r requirements-infer.txt
    echo.
    pause
    exit /b 1
)

REM One process now — server.py serves the ui/ page itself (same origin, no
REM CORS dance, no separate static-file-server window). Still launched in
REM its own titled window (not this one) so the browser can be opened only
REM once it's actually listening, same ordering guarantee the old two-window
REM version had.
echo Starting the server ^(serves both the API and the demo page^) ...
start "Jugalbandi Server (port 7860)" /D "%~dp0" cmd /k ""%~dp0venv\Scripts\python.exe" openenv_server\server.py"

timeout /t 3 >nul
start http://localhost:7860

echo.
echo Server is running in the "Jugalbandi Server" window - its logs (and the
echo first /infer call's one-time model-load pause, up to a minute or two on
echo CPU) are there, not here. Close that window when you're done.
echo This window can be closed now.
echo.
pause >nul
