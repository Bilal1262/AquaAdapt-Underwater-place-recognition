#!/usr/bin/env bash
set -euo pipefail

config="configs/mclab12_train_5hz.yaml"
latest="/home/bb/Aqua_adapt/artifacts/checkpoints/mclab12_train_5hz/projection_head_only/latest.pt"
completion="/home/bb/Aqua_adapt/artifacts/checkpoints/mclab12_train_5hz/projection_head_only/training_complete.json"

aquaadapt build-manifest --config "$config"
if [[ -f "$completion" ]]; then
  echo "MCLab 1+2 training is already complete: $completion"
  exit 0
elif [[ -f "$latest" ]]; then
  echo "Resuming MCLab 1+2 training from $latest"
  aquaadapt train \
    --config "$config" \
    --mode projection_head_only \
    --resume "$latest"
else
  aquaadapt train \
    --config "$config" \
    --mode projection_head_only
fi
