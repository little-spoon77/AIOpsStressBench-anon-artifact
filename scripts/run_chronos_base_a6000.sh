#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/hsc2_foundation"
export PYTHONPATH="$HOME/hsc2_foundation"
export HF_HOME="$HOME/hsc2_foundation/.hf_cache"
export HF_HUB_CACHE="$HOME/hsc2_foundation/.hf_cache/hub"
unset TRANSFORMERS_CACHE
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TMPDIR="$HOME/hsc2_foundation/tmp"
export CUDA_VISIBLE_DEVICES=1

PY="$HOME/hsc2_foundation/.venv_lwx/bin/python"
OUT="outputs/foundation_reference"
mkdir -p "$OUT"

SCENARIOS=(clean missing_30 missing_variables_30 delayed_12)

while read -r cfg source dataset; do
  echo "[RUN] chronos-bolt-base ${dataset}"
  "$PY" scripts/run_chronos_reference.py \
    --model-id "autogluon/chronos-bolt-base" \
    --base-config "configs/${cfg}" \
    --summary "${OUT}/chronos_bolt_base_${dataset}_a6000.csv" \
    --source "${source}" \
    --dataset "${dataset}" \
    --scenarios "${SCENARIOS[@]}" \
    --max-windows 512 \
    --batch-size 16 \
    --device cuda \
    --dtype bfloat16 \
    --latency-warmup 3 \
    --latency-iters 10
done <<'EOF'
alibaba2018_machine_usage.yaml alibaba2018 alibaba2018
salesforce_borg_256x2048.yaml salesforce_borg salesforce_borg
EOF
