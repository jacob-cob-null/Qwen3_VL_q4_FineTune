# pAIge environment setup — Python 3.11, CUDA 12.1 (RTX 3060 / 12 GB VRAM)
# Usage: powershell -ExecutionPolicy Bypass -File setup_env.ps1

$ErrorActionPreference = "Stop"
$VENV = ".venv311"
$PY   = "$VENV\Scripts\python.exe"

Write-Host "=== pAIge Environment Setup ===" -ForegroundColor Cyan

# 1. Virtual environment
if (Test-Path $VENV) {
    Write-Host "[1/4] Venv already exists — skipping creation." -ForegroundColor Gray
} else {
    Write-Host "[1/4] Creating virtual environment (.venv311)..." -ForegroundColor Yellow
    python -m venv $VENV
    Write-Host "      Done." -ForegroundColor Green
}

# 2. Upgrade pip
Write-Host "[2/4] Upgrading pip..." -ForegroundColor Yellow
& $PY -m pip install --upgrade pip setuptools wheel | Out-Null

# 3. PyTorch 2.5.1 + CUDA 12.1  (must precede everything else)
Write-Host "[3/4] Installing PyTorch 2.5.1+cu121..." -ForegroundColor Yellow
& $PY -m pip install `
    torch==2.5.1+cu121 `
    torchvision==0.20.1+cu121 `
    torchaudio==2.5.1+cu121 `
    --index-url https://download.pytorch.org/whl/cu121
Write-Host "      Done." -ForegroundColor Green

# 4. All remaining packages (fine-tuning stack + utilities)
Write-Host "[4/4] Installing packages from requirements.txt..." -ForegroundColor Yellow
& $PY -m pip install -r requirements.txt
Write-Host "      Done." -ForegroundColor Green

Write-Host ""
Write-Host "=== Setup complete ===" -ForegroundColor Cyan
Write-Host "Activate : .\.venv311\Scripts\Activate.ps1"
Write-Host "Verify   : python verify_gpu.py"
