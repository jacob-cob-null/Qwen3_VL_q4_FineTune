# PowerShell environment setup for pAIge (RTX 3060, Windows)
# Run: powershell -ExecutionPolicy Bypass -File setup_paige_env.ps1

Write-Host "=== pAIge Environment Setup for RTX 3060 ===" -ForegroundColor Cyan
Write-Host ""

# Step 1: Create virtual environment
Write-Host "[1/6] Creating virtual environment..." -ForegroundColor Yellow
if (Test-Path "paige_venv") {
    Write-Host "Virtual environment already exists. Skipping creation." -ForegroundColor Gray
} else {
    python -m venv paige_venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to create virtual environment" -ForegroundColor Red
        exit 1
    }
    Write-Host "Virtual environment created successfully" -ForegroundColor Green
}

# Step 2: Setup explicit Python path
Write-Host "[2/6] Setting up Explicit Python Path..." -ForegroundColor Yellow
$PY = ".\paige_venv\Scripts\python.exe"
if (-Not (Test-Path $PY)) {
    Write-Host "ERROR: Virtual environment Python not found at $PY" -ForegroundColor Red
    exit 1
}

# Step 3: Upgrade pip, setuptools, wheel
Write-Host "[3/6] Upgrading pip, setuptools, wheel..." -ForegroundColor Yellow
& $PY -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: pip upgrade had issues, continuing..." -ForegroundColor Yellow
}

# Step 4: Install PyTorch with CUDA 12.1 (CRITICAL - must be first)
Write-Host "[4/6] Installing PyTorch with CUDA 12.1 support..." -ForegroundColor Yellow
Write-Host "This may take several minutes..." -ForegroundColor Gray
& $PY -m pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: PyTorch installation failed" -ForegroundColor Red
    exit 1
}
Write-Host "PyTorch installed successfully" -ForegroundColor Green

# Step 5: Install core dependencies (NO torch reinstall)
Write-Host "[5/6] Installing core dependencies..." -ForegroundColor Yellow
& $PY -m pip install --no-deps transformers peft accelerate trl
& $PY -m pip install bitsandbytes datasets Pillow faker reportlab evaluate scikit-learn matplotlib --extra-index-url https://download.pytorch.org/whl/cu121
& $PY -m pip install numpy scipy --extra-index-url https://download.pytorch.org/whl/cu121
if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: Some packages had issues, but continuing..." -ForegroundColor Yellow
}

# Step 6: Install Unsloth (separate to avoid conflicts)
Write-Host "[6/6] Installing Unsloth..." -ForegroundColor Yellow
Write-Host "Attempting Unsloth installation (may fail - fallback to PEFT is available)..." -ForegroundColor Gray
& $PY -m pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" --extra-index-url https://download.pytorch.org/whl/cu121
if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: Unsloth installation failed - will use PEFT fallback during training" -ForegroundColor Yellow
} else {
    Write-Host "Unsloth installed successfully" -ForegroundColor Green
}

Write-Host ""
Write-Host "=== Setup Complete ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Green
Write-Host "1. Activate environment: .\paige_venv\Scripts\Activate.ps1"
Write-Host "2. Verify GPU: python verify_gpu.py"
Write-Host "3. Run pre-flight checks before training"
Write-Host ""