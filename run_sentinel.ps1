# BORDER SENTINEL — PowerShell Master Production Launcher
# 100% Air-Gapped / Zero Internet Connection Required

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "   BORDER SENTINEL // AUTONOMOUS DEFENSE & INTELLIGENCE GRID" -ForegroundColor Cyan
Write-Host "   100% Localhost Architecture | RTX 4060 GPU Accelerated" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

# 1. Virtual Environment Check
$VenvPy = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPy)) {
    Write-Host "[ERROR] Virtual environment (.venv) not found!" -ForegroundColor Red
    exit 1
}
Write-Host "[✓] Python 3.11 Virtual Environment verified: $VenvPy" -ForegroundColor Green

# 2. Local Weights Check
$ModelV2 = Join-Path $Root "models\registry\YOLO-L-v002\weights\best.pt"
$ModelBase = Join-Path $Root "yolov8l.pt"
if (Test-Path $ModelV2) {
    Write-Host "[✓] Custom Fine-Tuned Weights verified: YOLO-L-v002" -ForegroundColor Green
} elseif (Test-Path $ModelBase) {
    Write-Host "[✓] Base YOLOv8l Weights verified: $ModelBase" -ForegroundColor Yellow
} else {
    Write-Host "[ERROR] No model weights found!" -ForegroundColor Red
    exit 1
}

# 3. Clean Port 8000
Write-Host "[*] Checking port 8000 availability..." -ForegroundColor Gray
$connections = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue
if ($connections) {
    foreach ($conn in $connections) {
        $pidToKill = $conn.OwningProcess
        if ($pidToKill -gt 0) {
            Write-Host "[*] Terminating lingering process $pidToKill on port 8000..." -ForegroundColor Yellow
            Stop-Process -Id $pidToKill -Force -ErrorAction SilentlyContinue
        }
    }
}

# 4. Launch Browser
Write-Host "[✓] Starting browser at http://127.0.0.1:8000" -ForegroundColor Green
Start-Process "http://127.0.0.1:8000"

# 5. Start Server
Write-Host ""
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "TACTICAL GRID ACTIVE — Press CTRL+C to terminate." -ForegroundColor Cyan
Write-Host "API Explorer:    http://127.0.0.1:8000/docs" -ForegroundColor Gray
Write-Host "AI Studio:       http://127.0.0.1:8000/annotate" -ForegroundColor Gray
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""

& $VenvPy -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
