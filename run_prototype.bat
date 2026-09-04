@echo off
TITLE Offline Surveillance & Defense Grid - Prototype v0.1
echo ======================================================================
echo STARTING OFFLINE SURVEILLANCE & DEFENSE GRID (PROTOTYPE v0.1)
echo 100%% Local Architecture - Zero Internet Connection Required
echo ======================================================================

REM 1. Activate virtual environment
IF NOT EXIST ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found! Run setup instructions first.
    pause
    exit /b 1
)
call .venv\Scripts\activate.bat

REM 2. Verify model weights exist locally
IF NOT EXIST "ai\models\yolov8l.pt" (
    echo [ERROR] Model weights ai\models\yolov8l.pt missing!
    pause
    exit /b 1
)

REM 3. Launch browser to local dashboard after 2 seconds
start "" http://127.0.0.1:8000

REM 4. Start local backend server
echo [System] Server running at http://127.0.0.1:8000
echo [System] Press CTRL+C to terminate the prototype.
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
