#!/usr/bin/env bash
set -euo pipefail

config="configs/mclab3_heldout_5hz.yaml"
checkpoint="/home/bb/Aqua_adapt/artifacts/checkpoints/mclab12_train_5hz/projection_head_only/best.pt"

# This script never trains on MCLab 3.
aquaadapt extract --config "$config"
aquaadapt parse-trajectory --config "$config"
aquaadapt build-manifest --config "$config"
aquaadapt baseline --config "$config" --method raw_dinov2
aquaadapt encode --config "$config" --checkpoint "$checkpoint" --method aquaadapt
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
