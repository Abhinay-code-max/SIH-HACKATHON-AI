@echo off
TITLE BORDER SENTINEL -- Master Grand Demonstration Launcher
color 0B
echo ======================================================================
echo    BORDER SENTINEL // ONE-CLICK MASTER DEMONSTRATION
echo    100%% Air-Gapped / Zero Internet Required / RTX 4060 Accelerated
echo ======================================================================
echo.
REM 1. Virtual Environment Check
if exist .venv\Scripts\activate.bat goto VENV_OK
echo [ERROR] Python virtual environment not found!
echo Please ensure .venv is installed.
pause
exit /b 1
:VENV_OK
REM 2. Local Weights Check
if exist training_lab\runs\U2_yolov8m_640_9class_v002\weights\best.pt (
    echo [+] Verified Primary 9-Class Detector: U2 (training_lab\runs\U2_yolov8m_640_9class_v002\weights\best.pt)
    goto WEIGHTS_OK
)
if exist models\registry\YOLO-L-v002\weights\best.pt goto WEIGHTS_OK
if exist ai\models\yolov8l.pt goto WEIGHTS_OK
if exist yolov8l.pt goto WEIGHTS_OK
echo [ERROR] U2 9-class model checkpoint is required at training_lab\runs\U2_yolov8m_640_9class_v002\weights\best.pt!
pause
exit /b 1
:WEIGHTS_OK
echo [+] Pre-flight verification passed: Python venv and AI weights ready.
REM 3. Port 8000 Cleanup
echo [*] Terminating any existing processes on port 8000...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000" ^| findstr "LISTENING"') do taskkill /F /PID %%a >nul 2>&1
REM 4. Launch Server in Companion Window
echo [*] Starting BORDER SENTINEL Grid Server in background...
start "BORDER SENTINEL GRID SERVER" cmd /c "call .venv\Scripts\activate.bat && python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000"
REM 5. Wait for Server Initialization
echo [*] Initializing tactical engine and loading local GPU weights...
timeout /t 4 >nul
REM 6. Launch Tactical Command Dashboard
echo [+] Opening Command Dashboard at http://127.0.0.1:8000
start "" http://127.0.0.1:8000
REM 7. Give Browser 2 Seconds to Render
timeout /t 2 >nul
echo.
echo ======================================================================
echo INJECTING LIVE TACTICAL DEFENSE SCENARIOS (AUTOMATED DEMO)
echo Watch the dashboard in your browser to observe real-time DEFCON shifts,
echo geofence alerts, tripwire crossings, and cross-camera Re-ID handoffs!
echo ======================================================================
echo.
REM 8. Run Demo Scenario Injector
call .venv\Scripts\activate.bat
python tools\demo_scenario_injector.py --auto
echo.
echo ======================================================================
echo GRAND DEMONSTRATION COMPLETE
echo Press [1] to Re-run Scenarios
echo Press [2] to Stop All Servers and Exit
echo ======================================================================
:MENU
set /p opt="Select an option [1 or 2]: "
if "%opt%"=="1" goto RE_RUN
if "%opt%"=="2" goto STOP_ALL
goto MENU
:RE_RUN
cls
python tools\demo_scenario_injector.py --auto
goto MENU
:STOP_ALL
echo [*] Shutting down tactical grid...
call stop_sentinel.bat
exit /b 0
