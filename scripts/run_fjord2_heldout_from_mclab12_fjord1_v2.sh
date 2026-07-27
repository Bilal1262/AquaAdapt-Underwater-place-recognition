#!/usr/bin/env bash
set -euo pipefail

config="configs/fjord2_heldout_from_mclab12_fjord1_v2.yaml"
train_root="/home/bb/Aqua_adapt/artifacts/checkpoints/mclab12_fjord1_train_v2/projection_head_only"
checkpoint="$train_root/best.pt"
completion="$train_root/training_complete.json"
bag="/mnt/windows/datasets/ntnu_underwater/subset-fjord/fjord_2/fjord_2.bag"
tum="/mnt/windows/datasets/ntnu_underwater/subset-fjord/fjord_2/fjord_2_baseline.tum"
result_root="results/fjord2_heldout_from_mclab12_fjord1_v2"

if [[ ! -f "$checkpoint" || ! -f "$completion" ]]; then
  echo "Three-trajectory AquaAdapt V2 training is not complete yet." >&2
  echo "Finish it first with: bash scripts/run_mclab12_fjord1_train_v2.sh" >&2
  exit 2
fi

if [[ ! -f "$bag" ]]; then
  echo "Missing Fjord2 bag: $bag" >&2
  exit 2
fi

if [[ ! -f "$tum" ]]; then
  echo "Missing Fjord2 reference trajectory: $tum" >&2
  exit 2
fi

echo "Preparing held-out Fjord2 frames and poses at 5 Hz."
aquaadapt extract --config "$config"
aquaadapt parse-trajectory --config "$config"
aquaadapt build-manifest --config "$config"

echo "Encoding raw DINOv2 and AquaAdapt descriptors."
aquaadapt baseline --config "$config" --method raw_dinov2
aquaadapt encode \
  --config "$config" \
  --checkpoint "$checkpoint" \
  --method aquaadapt

echo "Evaluating clean held-out retrieval."
aquaadapt evaluate --config "$config" --method raw_dinov2
aquaadapt evaluate --config "$config" --method aquaadapt

echo "Evaluating corruption robustness."
aquaadapt robustness \
  --config "$config" \
  --checkpoint "$checkpoint" \
  --methods raw_dinov2,aquaadapt

echo "Generating qualitative retrieval examples and report."
aquaadapt visualize-retrievals \
  --config "$config" \
  --checkpoint "$checkpoint" \
  --queries 12 \
  --severity 2
aquaadapt report --config "$config"

echo "Held-out Fjord2 evaluation complete."
echo "Report: $result_root/report.md"
echo "Gallery: $result_root/qualitative/index.html"
