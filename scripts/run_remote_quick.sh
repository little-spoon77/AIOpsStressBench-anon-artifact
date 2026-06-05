#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -x ".conda-env/bin/python" ]; then
  PYTHON=".conda-env/bin/python"
else
  /opt/miniconda3/bin/conda create -y -p "$PWD/.conda-env" python=3.10 pip
  PYTHON=".conda-env/bin/python"
fi

"$PYTHON" -m pip install -r requirements.txt
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PYTHON" -m race_forecast.run --config configs/quick_synthetic.yaml
