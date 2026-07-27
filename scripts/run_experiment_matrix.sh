#!/usr/bin/env bash
set -uo pipefail

# One entry point for every training and held-out evaluation workflow currently
# included in AquaAdapt. Individual scripts remain responsible for checkpoint
# resume, dataset validation, descriptor caching, and report generation.

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
cd "$repo_root" || exit 2
data_root="${AQUAADAPT_DATA_ROOT:-/mnt/windows/datasets/ntnu_underwater}"
artifact_root="${AQUAADAPT_ARTIFACT_ROOT:-$repo_root/artifacts}"

declare -a training_order=(
  "mclab1_full"
  "mclab12_v1"
  "mclab12_v2"
  "mclab12_fjord1_v2"
)

declare -A training_script=(
  ["mclab1_full"]="scripts/run_full_pipeline.sh"
  ["mclab12_v1"]="scripts/run_mclab12_train.sh"
  ["mclab12_v2"]="scripts/run_mclab12_train_v2.sh"
  ["mclab12_fjord1_v2"]="scripts/run_mclab12_fjord1_train_v2.sh"
)

declare -A training_description=(
  ["mclab1_full"]="Train/evaluate the original single-trajectory MCLab1 pipeline"
  ["mclab12_v1"]="Train the original projection head on MCLab1 + MCLab2"
  ["mclab12_v2"]="Train the residual V2 adapter on MCLab1 + MCLab2"
  ["mclab12_fjord1_v2"]="Train the residual V2 adapter on MCLab1 + MCLab2 + Fjord1"
)

declare -A training_marker=(
  ["mclab1_full"]="$data_root/processed/checkpoints/mclab1_full/projection_head_only/training_complete.json"
  ["mclab12_v1"]="$artifact_root/checkpoints/mclab12_train_5hz/projection_head_only/training_complete.json"
  ["mclab12_v2"]="$artifact_root/checkpoints/mclab12_train_v2/projection_head_only/training_complete.json"
  ["mclab12_fjord1_v2"]="$artifact_root/checkpoints/mclab12_fjord1_train_v2/projection_head_only/training_complete.json"
)

declare -a evaluation_order=(
  "mclab1_to_mclab2"
  "mclab12_v1_to_mclab3"
  "mclab12_v1_to_fjord1"
  "mclab12_v2_to_fjord1"
  "mclab12_fjord1_v2_to_fjord2"
)

declare -A evaluation_script=(
  ["mclab1_to_mclab2"]="scripts/run_mclab2_transfer.sh"
  ["mclab12_v1_to_mclab3"]="scripts/run_mclab3_heldout.sh"
  ["mclab12_v1_to_fjord1"]="scripts/run_fjord1_heldout.sh"
  ["mclab12_v2_to_fjord1"]="scripts/run_fjord1_heldout_v2.sh"
  ["mclab12_fjord1_v2_to_fjord2"]="scripts/run_fjord2_heldout_from_mclab12_fjord1_v2.sh"
)

declare -A evaluation_description=(
  ["mclab1_to_mclab2"]="Evaluate the MCLab1 head on held-out MCLab2"
  ["mclab12_v1_to_mclab3"]="Evaluate the MCLab1+2 V1 head on held-out MCLab3"
  ["mclab12_v1_to_fjord1"]="Evaluate the MCLab1+2 V1 head on held-out Fjord1"
  ["mclab12_v2_to_fjord1"]="Evaluate the MCLab1+2 V2 adapter on held-out Fjord1"
  ["mclab12_fjord1_v2_to_fjord2"]="Evaluate the MCLab1+2+Fjord1 V2 adapter on held-out Fjord2"
)

declare -A evaluation_marker=(
  ["mclab1_to_mclab2"]="results/mclab2_transfer_5hz/report.md"
  ["mclab12_v1_to_mclab3"]="results/mclab3_heldout_from_mclab12/report.md"
  ["mclab12_v1_to_fjord1"]="results/fjord1_heldout_from_mclab12/report.md"
  ["mclab12_v2_to_fjord1"]="results/fjord1_heldout_from_mclab12_v2/report.md"
  ["mclab12_fjord1_v2_to_fjord2"]="results/fjord2_heldout_from_mclab12_fjord1_v2/report.md"
)

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_experiment_matrix.sh --list
  bash scripts/run_experiment_matrix.sh --recommended [options]
  bash scripts/run_experiment_matrix.sh --all [options]
  bash scripts/run_experiment_matrix.sh --train ID[,ID...] [--eval ID[,ID...]] [options]

Selection:
  --list                 List every registered training/evaluation combination.
  --recommended          Run the final three-dataset V2 training and Fjord2 evaluation.
  --all                  Run every registered training, then every evaluation.
  --train ID[,ID...]     Select training jobs. Use "all" for every training job.
  --eval ID[,ID...]      Select evaluation jobs. Use "all" for every evaluation job.
  --smoke                Run the bounded quick pipeline before selected jobs.

Execution:
  --dry-run              Print the execution plan without running commands.
  --skip-completed       Skip jobs whose completion marker/report already exists.
  --keep-going           Continue after a failed job and report all failures at the end.
  -h, --help             Show this help.

Examples:
  bash scripts/run_experiment_matrix.sh --recommended
  bash scripts/run_experiment_matrix.sh --train mclab12_v2 --eval mclab12_v2_to_fjord1
  bash scripts/run_experiment_matrix.sh --all --skip-completed --keep-going
  bash scripts/run_experiment_matrix.sh --all --dry-run

Logs are written under artifacts/logs/experiment_matrix/<timestamp>/.
EOF
}

list_jobs() {
  echo "TRAINING COMBINATIONS"
  for id in "${training_order[@]}"; do
    printf '  %-34s %s\n' "$id" "${training_description[$id]}"
  done
  echo
  echo "EVALUATION COMBINATIONS"
  for id in "${evaluation_order[@]}"; do
    printf '  %-34s %s\n' "$id" "${evaluation_description[$id]}"
  done
}

append_unique() {
  local -n target=$1
  local candidate=$2
  local existing
  for existing in "${target[@]:-}"; do
    [[ "$existing" == "$candidate" ]] && return 0
  done
  target+=("$candidate")
}

add_selection() {
  local kind=$1
  local value=$2
  local target_name=$3
  local -a available=()
  local id

  if [[ "$kind" == "training" ]]; then
    available=("${training_order[@]}")
  else
    available=("${evaluation_order[@]}")
  fi

  if [[ "$value" == "all" ]]; then
    for id in "${available[@]}"; do
      append_unique "$target_name" "$id"
    done
    return 0
  fi

  local -a requested=()
  IFS=',' read -r -a requested <<< "$value"
  for id in "${requested[@]}"; do
    if [[ "$kind" == "training" && -z "${training_script[$id]+x}" ]]; then
      echo "Unknown training ID: $id" >&2
      return 2
    fi
    if [[ "$kind" == "evaluation" && -z "${evaluation_script[$id]+x}" ]]; then
      echo "Unknown evaluation ID: $id" >&2
      return 2
    fi
    append_unique "$target_name" "$id"
  done
}

declare -a selected_training=()
declare -a selected_evaluation=()
dry_run=false
skip_completed=false
keep_going=false
run_smoke=false
show_list=false

if [[ $# -eq 0 ]]; then
  usage
  exit 2
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --list)
      show_list=true
      shift
      ;;
    --recommended)
      append_unique selected_training "mclab12_fjord1_v2"
      append_unique selected_evaluation "mclab12_fjord1_v2_to_fjord2"
      shift
      ;;
    --all)
      add_selection training all selected_training || exit $?
      add_selection evaluation all selected_evaluation || exit $?
      shift
      ;;
    --train)
      [[ $# -ge 2 ]] || { echo "--train requires an ID or comma-separated IDs" >&2; exit 2; }
      add_selection training "$2" selected_training || exit $?
      shift 2
      ;;
    --eval)
      [[ $# -ge 2 ]] || { echo "--eval requires an ID or comma-separated IDs" >&2; exit 2; }
      add_selection evaluation "$2" selected_evaluation || exit $?
      shift 2
      ;;
    --smoke)
      run_smoke=true
      shift
      ;;
    --dry-run)
      dry_run=true
      shift
      ;;
    --skip-completed)
      skip_completed=true
      shift
      ;;
    --keep-going)
      keep_going=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$show_list" == true ]]; then
  list_jobs
  exit 0
fi

if [[ ${#selected_training[@]} -eq 0 && ${#selected_evaluation[@]} -eq 0 && "$run_smoke" == false ]]; then
  echo "No jobs selected." >&2
  usage >&2
  exit 2
fi

run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
log_root="${AQUAADAPT_LOG_ROOT:-artifacts/logs/experiment_matrix/$run_stamp}"
declare -a failures=()

run_job() {
  local kind=$1
  local id=$2
  local command_path=$3
  local marker=$4

  if [[ "$skip_completed" == true && -f "$marker" ]]; then
    printf '[SKIP] %s/%s — completion marker exists: %s\n' "$kind" "$id" "$marker"
    return 0
  fi

  if [[ ! -f "$command_path" ]]; then
    echo "[FAIL] $kind/$id — missing script: $command_path" >&2
    failures+=("$kind/$id")
    return 1
  fi

  if [[ "$dry_run" == true ]]; then
    printf '[DRY RUN] %-10s %-34s bash %s\n' "$kind" "$id" "$command_path"
    return 0
  fi

  mkdir -p "$log_root"
  local log_file="$log_root/${kind}_${id}.log"
  echo
  echo "================================================================"
  echo "Running $kind/$id"
  echo "Command: bash $command_path"
  echo "Log: $log_file"
  echo "================================================================"

  bash "$command_path" 2>&1 | tee "$log_file"
  local status=${PIPESTATUS[0]}
  if [[ $status -ne 0 ]]; then
    echo "[FAIL] $kind/$id exited with status $status" >&2
    failures+=("$kind/$id")
    return "$status"
  fi

  echo "[PASS] $kind/$id"
  return 0
}

if [[ "$run_smoke" == true ]]; then
  if ! run_job smoke quick scripts/run_quick_pipeline.sh \
    "/mnt/windows/datasets/ntnu_underwater/processed/checkpoints/mclab1_quick/projection_head_only/training_complete.json"; then
    [[ "$keep_going" == true ]] || exit 1
  fi
fi

for id in "${selected_training[@]}"; do
  if ! run_job training "$id" "${training_script[$id]}" "${training_marker[$id]}"; then
    [[ "$keep_going" == true ]] || exit 1
  fi
done

for id in "${selected_evaluation[@]}"; do
  if ! run_job evaluation "$id" "${evaluation_script[$id]}" "${evaluation_marker[$id]}"; then
    [[ "$keep_going" == true ]] || exit 1
  fi
done

echo
if [[ ${#failures[@]} -gt 0 ]]; then
  echo "Experiment matrix completed with failures:"
  printf '  - %s\n' "${failures[@]}"
  echo "Logs: $log_root"
  exit 1
fi

if [[ "$dry_run" == true ]]; then
  echo "Dry run complete; no commands were executed."
else
  echo "All selected jobs completed successfully."
  echo "Logs: $log_root"
fi
