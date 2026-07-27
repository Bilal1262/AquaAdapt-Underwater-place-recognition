#!/usr/bin/env bash
set -euo pipefail

CONFIG="${AQUAADAPT_CONFIG:-configs/quick.yaml}"

aquaadapt doctor --config "$CONFIG"
aquaadapt inspect-bag --config "$CONFIG"
aquaadapt extract --config "$CONFIG" --quick
aquaadapt parse-trajectory --config "$CONFIG" --quick
aquaadapt build-manifest --config "$CONFIG" --quick
aquaadapt visualize-augmentations --config "$CONFIG" --quick
aquaadapt check-backbone --config "$CONFIG" --quick
aquaadapt baseline --config "$CONFIG" --method raw_dinov2 --quick
aquaadapt evaluate --config "$CONFIG" --method raw_dinov2 --quick
aquaadapt train --config "$CONFIG" --mode projection_head_only --quick

CHECKPOINT="/mnt/windows/datasets/ntnu_underwater/processed/checkpoints/mclab1_quick/projection_head_only/best.pt"
aquaadapt encode --config "$CONFIG" --checkpoint "$CHECKPOINT" --method aquaadapt --quick
aquaadapt evaluate --config "$CONFIG" --method aquaadapt --quick
aquaadapt visualize-retrievals --config "$CONFIG" --checkpoint "$CHECKPOINT" --quick --queries 6
aquaadapt robustness --config "$CONFIG" --checkpoint "$CHECKPOINT" --methods raw_dinov2,aquaadapt --quick --limit 60
aquaadapt report --config "$CONFIG" --quick
