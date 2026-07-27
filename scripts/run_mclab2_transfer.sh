#!/usr/bin/env bash
set -euo pipefail

CONFIG="${AQUAADAPT_CONFIG:-configs/mclab2_transfer_5hz.yaml}"
CHECKPOINT="${AQUAADAPT_CHECKPOINT:-/mnt/windows/datasets/ntnu_underwater/processed/checkpoints/mclab1_5hz_head/projection_head_only/best.pt}"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Missing mclab_1 checkpoint: $CHECKPOINT" >&2
  exit 2
fi

# mclab_2 is evaluation-only: never invoke `aquaadapt train` in this script.
aquaadapt extract --config "$CONFIG"
aquaadapt parse-trajectory --config "$CONFIG"
aquaadapt build-manifest --config "$CONFIG"

aquaadapt baseline --config "$CONFIG" --method raw_dinov2
aquaadapt encode --config "$CONFIG" --checkpoint "$CHECKPOINT" --method aquaadapt

aquaadapt evaluate --config "$CONFIG" --method raw_dinov2
aquaadapt evaluate --config "$CONFIG" --method aquaadapt

aquaadapt visualize-retrievals \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --queries 12 \
  --severity 2

aquaadapt report --config "$CONFIG"

echo "Transfer report: results/mclab2_transfer_5hz/report.md"
echo "Visual gallery: results/mclab2_transfer_5hz/qualitative/index.html"
