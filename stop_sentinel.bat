@echo off
TITLE Stop BORDER SENTINEL Server
echo [*] Terminating BORDER SENTINEL processes on port 8000...

for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000" ^| findstr "LISTENING"') do (
    echo [*] Killing process PID %%a...
    taskkill /F /PID %%a >nul 2>&1
)

echo [✓] BORDER SENTINEL server stopped cleanly.
timeout /t 2 >nul
