#!/usr/bin/env bash
set -euo pipefail

config="configs/fjord1_heldout_v2.yaml"
train_root="/home/bb/Aqua_adapt/artifacts/checkpoints/mclab12_train_v2/projection_head_only"
checkpoint="$train_root/best.pt"
completion="$train_root/training_complete.json"
source_root="/mnt/windows/datasets/ntnu_underwater/processed/fjord_1/alphasense_driver_ros_cam0"
local_root="/home/bb/Aqua_adapt/artifacts/fjord_eval_v2/fjord_1/alphasense_driver_ros_cam0"

if [[ ! -f "$checkpoint" || ! -f "$completion" ]]; then
  echo "AquaAdapt V2 training is not complete yet." >&2
  echo "Finish it first with: bash scripts/run_mclab12_train_v2.sh" >&2
  exit 2
fi

if [[ ! -f "$source_root/metadata.csv" ]]; then
  echo "Missing prepared Fjord metadata: $source_root/metadata.csv" >&2
  exit 2
fi

# Copy metadata only. Images remain immutable on the dataset mount.
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

echo "V2 Fjord evaluation complete."
echo "Report: results/fjord1_heldout_from_mclab12_v2/report.md"
echo "Gallery: results/fjord1_heldout_from_mclab12_v2/qualitative/index.html"
