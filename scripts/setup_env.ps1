# Windows PowerShell setup. Run from the project root:  powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
# Optional: -Cuda cpu   to install CPU-only PyTorch.
param([string]$Cuda = "cu128")
$ErrorActionPreference = "Stop"
if (-not (Test-Path .venv)) { python -m venv .venv }
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/$Cuda"
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m playwright install chromium
.\.venv\Scripts\python scripts\setup_weights.py
Push-Location frontend; npm install; Pop-Location
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
.\.venv\Scripts\python -c "import torch; print('torch', torch.__version__, 'CUDA', torch.cuda.is_available())"
Write-Host "Setup complete. Next: extract DAWN into data\DAWN (see docs\DATASETS.md) and run scripts\validate_datasets.py"
