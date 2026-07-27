#!/usr/bin/env bash
set -euo pipefail

CONFIG="${AQUAADAPT_CONFIG:-configs/full.yaml}"

aquaadapt doctor --config "$CONFIG"
aquaadapt inspect-bag --config "$CONFIG"
aquaadapt extract --config "$CONFIG"
aquaadapt parse-trajectory --config "$CONFIG"
aquaadapt build-manifest --config "$CONFIG"
aquaadapt visualize-augmentations --config "$CONFIG"
aquaadapt check-backbone --config "$CONFIG"
aquaadapt baseline --config "$CONFIG" --method raw_dinov2
aquaadapt baseline --config "$CONFIG" --method enhanced_dinov2
aquaadapt train --config "$CONFIG" --mode projection_head_only

CHECKPOINT="/mnt/windows/datasets/ntnu_underwater/processed/checkpoints/mclab1_full/projection_head_only/best.pt"
aquaadapt encode --config "$CONFIG" --checkpoint "$CHECKPOINT" --method aquaadapt
for method in raw_dinov2 enhanced_dinov2 aquaadapt; do
  aquaadapt evaluate --config "$CONFIG" --method "$method"
done
aquaadapt robustness --config "$CONFIG" --checkpoint "$CHECKPOINT"
aquaadapt ablate --config "$CONFIG"
aquaadapt benchmark --config "$CONFIG" --checkpoint "$CHECKPOINT"
aquaadapt report --config "$CONFIG"

