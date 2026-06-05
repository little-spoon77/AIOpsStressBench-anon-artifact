#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-0}"
MAX_USED_MB="${MAX_USED_MB:-1024}"
MAX_UTIL="${MAX_UTIL:-20}"
CHECK_INTERVAL="${CHECK_INTERVAL:-60}"
MAX_CHECKS="${MAX_CHECKS:-1}"

if [ "$#" -lt 1 ]; then
  echo "Usage: GPU_ID=0 MAX_USED_MB=1024 MAX_UTIL=20 $0 <command> [args...]" >&2
  exit 2
fi

check_gpu() {
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i "$GPU_ID" |
    awk -F',' -v max_mem="$MAX_USED_MB" -v max_util="$MAX_UTIL" '{
      gsub(/ /, "", $1);
      gsub(/ /, "", $2);
      if ($1 <= max_mem && $2 <= max_util) {
        exit 0;
      }
      exit 1;
    }'
}

for ((attempt = 1; attempt <= MAX_CHECKS; attempt++)); do
  if check_gpu; then
    echo "GPU ${GPU_ID} is free enough; launching command."
    exec env CUDA_VISIBLE_DEVICES="$GPU_ID" "$@"
  fi
  echo "GPU ${GPU_ID} is busy; attempt ${attempt}/${MAX_CHECKS}."
  if [ "$attempt" -lt "$MAX_CHECKS" ]; then
    sleep "$CHECK_INTERVAL"
  fi
done

echo "GPU ${GPU_ID} did not become free under thresholds memory<=${MAX_USED_MB}MB util<=${MAX_UTIL}%." >&2
exit 75
