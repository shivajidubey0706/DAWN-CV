#!/usr/bin/env bash
# Linux/macOS setup. Run from the project root:  bash scripts/setup_env.sh [cu128|cpu]
set -euo pipefail
CUDA="${1:-cu128}"
[ -d .venv ] || python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
if [ "$(uname)" = "Darwin" ]; then pip install torch torchvision; else pip install torch torchvision --index-url "https://download.pytorch.org/whl/${CUDA}"; fi
pip install -r requirements.txt
python -m playwright install chromium
python scripts/setup_weights.py
(cd frontend && npm install)
[ -f .env ] || cp .env.example .env
python -c "import torch; print('torch', torch.__version__, 'CUDA', torch.cuda.is_available())"
echo "Setup complete. Next: extract DAWN into data/DAWN (see docs/DATASETS.md) and run scripts/validate_datasets.py"
