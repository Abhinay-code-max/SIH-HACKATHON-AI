@echo off
TITLE BORDER SENTINEL -- Autonomous Defense and Multi-Camera Intelligence Grid
color 0B

echo ======================================================================
echo    BORDER SENTINEL // AUTONOMOUS DEFENSE AND INTELLIGENCE GRID
echo    100%% Air-Gapped / Zero Internet Required / Localhost Architecture
echo ======================================================================
echo.

REM 1. Verify Virtual Environment
IF NOT EXIST ".venv\Scripts\activate.bat" (
    echo [ERROR] Python virtual environment was not found!
    echo Please run: py -3.11 -m venv .venv
    pause
    exit /b 1
)
call .venv\Scripts\activate.bat

REM 2. Verify AI Weights
IF EXIST "training_lab\runs\U2_yolov8m_640_9class_v002\weights\best.pt" (
    echo [System] Primary 9-Class Detector verified: U2 (training_lab\runs\U2_yolov8m_640_9class_v002\weights\best.pt)
) ELSE IF EXIST "models\registry\YOLO-L-v002\weights\best.pt" (
    echo [System] AI Model Weights verified: models\registry\YOLO-L-v002\weights\best.pt
) ELSE IF EXIST "ai\models\yolov8l.pt" (
    echo [System] AI Model Weights verified: ai\models\yolov8l.pt
) ELSE IF EXIST "yolov8l.pt" (
    echo [System] AI Model Weights verified: yolov8l.pt
) ELSE (
    echo [ERROR] U2 9-class model checkpoint is required at training_lab\runs\U2_yolov8m_640_9class_v002\weights\best.pt!
    pause
    exit /b 1
)

REM 2b. Verify Demo Surveillance Videos
IF NOT EXIST "training_lab\videos\CAM_01_gateway.mp4" (
    echo [System] Generating missing 5-camera surveillance demo videos...
    python tools\generate_sample_videos.py
)
echo [System] Clearing any lingering processes on port 8000...

for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000" ^| findstr "LISTENING"') do (
    taskkill /F /PID %%a >nul 2>&1
)

echo [System] Launching BORDER SENTINEL Dashboard at http://127.0.0.1:8000
start "" http://127.0.0.1:8000

echo.
echo ======================================================================
echo SERVER ACTIVE -- Press CTRL+C to terminate the tactical grid.
echo API Documentation: http://127.0.0.1:8000/docs
echo AI Studio:         http://127.0.0.1:8000/annotate
echo ======================================================================
echo.

python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
