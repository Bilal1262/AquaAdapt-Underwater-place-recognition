#!/usr/bin/env bash
set -euo pipefail

config="configs/mclab12_fjord1_train_v2.yaml"
run_root="/home/bb/Aqua_adapt/artifacts/checkpoints/mclab12_fjord1_train_v2/projection_head_only"
latest="$run_root/latest.pt"
completion="$run_root/training_complete.json"

required_manifests=(
  "/mnt/windows/datasets/ntnu_underwater/processed/mclab_1/alphasense_driver_ros_cam0/manifest.csv"
  "/mnt/windows/datasets/ntnu_underwater/processed/mclab_2/alphasense_driver_ros_cam0/manifest.csv"
  "/mnt/windows/datasets/ntnu_underwater/processed/fjord_1/alphasense_driver_ros_cam0/manifest.csv"
)

for manifest in "${required_manifests[@]}"; do
  if [[ ! -f "$manifest" ]]; then
    echo "Missing training manifest: $manifest" >&2
    echo "Prepare that trajectory before starting combined training." >&2
    exit 2
  fi
done

aquaadapt build-manifest --config "$config"

if [[ -f "$completion" ]]; then
  echo "Three-trajectory AquaAdapt V2 training is already complete: $completion"
  echo "Best checkpoint: $run_root/best.pt"
  exit 0
fi

if [[ -f "$latest" ]]; then
  echo "Resuming three-trajectory AquaAdapt V2 training from $latest"
  aquaadapt train \
    --config "$config" \
    --mode projection_head_only \
    --resume "$latest"
else
  echo "Starting AquaAdapt V2 training on MCLab 1 + MCLab 2 + Fjord 1"
  echo "Fjord 2 remains held out for final evaluation."
  aquaadapt train \
    --config "$config" \
    --mode projection_head_only
fi

echo "Training complete."
echo "Best checkpoint: $run_root/best.pt"
