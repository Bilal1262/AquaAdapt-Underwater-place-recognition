#!/usr/bin/env bash
set -euo pipefail

config="configs/mclab12_train_v2.yaml"
run_root="/home/bb/Aqua_adapt/artifacts/checkpoints/mclab12_train_v2/projection_head_only"
latest="$run_root/latest.pt"
completion="$run_root/training_complete.json"

aquaadapt build-manifest --config "$config"

if [[ -f "$completion" ]]; then
  echo "AquaAdapt V2 training is already complete: $completion"
  echo "Best checkpoint: $run_root/best.pt"
  exit 0
fi

if [[ -f "$latest" ]]; then
  echo "Resuming AquaAdapt V2 training from $latest"
  aquaadapt train \
    --config "$config" \
    --mode projection_head_only \
    --resume "$latest"
else
  echo "Starting AquaAdapt V2 training on MCLab 1 + MCLab 2"
  aquaadapt train \
    --config "$config" \
    --mode projection_head_only
fi

echo "Training complete."
echo "Best checkpoint: $run_root/best.pt"
