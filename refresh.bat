@echo off
REM ── NBA Prediction Dashboard refresh ────────────────────────────────────────
REM Double-click this file to:
REM   1. Re-fetch the latest NBA games from Basketball Reference
REM   2. Re-train the neural network
REM   3. Regenerate dashboard.html with the new predictions
REM   4. Open the dashboard in your browser
REM
REM First time? You may need to: pip install -r requirements.txt
REM ───────────────────────────────────────────────────────────────────────────

setlocal
cd /d "%~dp0"

echo.
echo ============================================================
echo   NBA Prediction Dashboard - Live Refresh
echo ============================================================
echo.

REM Activate virtualenv if present
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
    echo Activated virtual environment.
    echo.
)

REM Run the dashboard generator with --refresh and --open
python dashboard.py --refresh --open

if errorlevel 1 (
    echo.
    echo ============================================================
    echo   Refresh failed. Check error messages above.
    echo ============================================================
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   Done! Dashboard opened in your browser.
echo ============================================================
echo.
timeout /t 5
