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

echo Starting the server ^(this window^) and the demo page ^(new window^) ...
start "Jugalbandi UI (port 5500)" /D "%~dp0ui" cmd /k ""%~dp0venv\Scripts\python.exe" -m http.server 5500"

timeout /t 2 >nul
start http://localhost:5500

echo.
echo Server logs below. First real /infer call loads the model into memory
echo and can take a minute or two on CPU - that's normal, not stuck.
echo Close this window (or Ctrl+C) to stop the server when you're done.
echo.
venv\Scripts\python.exe openenv_server\server.py
