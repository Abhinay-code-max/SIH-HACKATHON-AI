@echo off
TITLE BORDER SENTINEL — Autonomous Defense & Multi-Camera Intelligence Grid
color 0B

echo ======================================================================
echo    BORDER SENTINEL // AUTONOMOUS DEFENSE & INTELLIGENCE GRID
echo    100%% Air-Gapped / Zero Internet Required / Localhost Architecture
echo ======================================================================
echo.

REM 1. Verify Virtual Environment
IF NOT EXIST ".venv\Scripts\activate.bat" (
    echo [ERROR] Python virtual environment (.venv) not found!
    echo Please run: py -3.11 -m venv .venv
    pause
    exit /b 1
)
call .venv\Scripts\activate.bat

REM 2. Verify AI Weights
IF NOT EXIST "models\registry\YOLO-L-v002\weights\best.pt" (
    IF NOT EXIST "yolov8l.pt" (
        echo [ERROR] No local YOLO weights found in models\registry or root directory!
        pause
        exit /b 1
    )
)

echo [System] Virtual environment activated (.venv)
echo [System] AI Model Weights verified (YOLO-L-v002)
echo [System] Clearing any lingering processes on port 8000...

for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000" ^| findstr "LISTENING"') do (
    taskkill /F /PID %%a >nul 2>&1
)

echo [System] Launching BORDER SENTINEL Dashboard at http://127.0.0.1:8000
start "" http://127.0.0.1:8000

echo.
echo ======================================================================
echo SERVER ACTIVE — Press CTRL+C to terminate the tactical grid.
echo API Documentation: http://127.0.0.1:8000/docs
echo AI Studio:         http://127.0.0.1:8000/annotate
echo ======================================================================
echo.

python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
