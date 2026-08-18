@echo off
echo Starting Autosub Player...

:: Check if the virtual environment exists
if not exist ".venv\Scripts\python.exe" (
    echo [INFO] First time setup: Creating virtual environment...
    python -m venv .venv
    
    echo [INFO] Installing dependencies...
    .\.venv\Scripts\pip install -r requirements.txt
)

:: Run the application
.\.venv\Scripts\python.exe autosub_player.py

:: Keep the window open if there's an error
if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] The application exited with an error.
    pause
)
