#!/usr/bin/env bash
set -euo pipefail

config="configs/fjord1_heldout_5hz.yaml"
checkpoint="/home/bb/Aqua_adapt/artifacts/checkpoints/mclab12_train_5hz/projection_head_only/best.pt"
completion="/home/bb/Aqua_adapt/artifacts/checkpoints/mclab12_train_5hz/projection_head_only/training_complete.json"
source_root="/mnt/windows/datasets/ntnu_underwater/processed/fjord_1/alphasense_driver_ros_cam0"
local_root="/home/bb/Aqua_adapt/artifacts/fjord_eval/fjord_1/alphasense_driver_ros_cam0"

if [[ ! -f "$checkpoint" || ! -f "$completion" ]]; then
  echo "MCLab 1+2 training is not complete yet." >&2
  echo "Finish training first with: bash scripts/run_mclab12_train.sh" >&2
  exit 2
fi

# Fjord 1 is strictly test-only: there is intentionally no training command.
if [[ ! -f "$source_root/metadata.csv" ]]; then
  echo "Missing prepared Fjord metadata: $source_root/metadata.csv" >&2
  exit 2
fi
mkdir -p "$local_root"
cp "$source_root/metadata.csv" "$local_root/metadata.csv"
cp "$source_root/extraction_summary.json" "$local_root/extraction_summary.json"

aquaadapt parse-trajectory --config "$config"
aquaadapt build-manifest --config "$config"
aquaadapt baseline --config "$config" --method raw_dinov2
aquaadapt encode \
  --config "$config" \
  --checkpoint "$checkpoint" \
  --method aquaadapt
aquaadapt evaluate --config "$config" --method raw_dinov2
aquaadapt evaluate --config "$config" --method aquaadapt
aquaadapt robustness \
  --config "$config" \
  --checkpoint "$checkpoint" \
  --methods raw_dinov2,aquaadapt
aquaadapt visualize-retrievals \
  --config "$config" \
  --checkpoint "$checkpoint" \
  --queries 12 \
  --severity 2
aquaadapt report --config "$config"
