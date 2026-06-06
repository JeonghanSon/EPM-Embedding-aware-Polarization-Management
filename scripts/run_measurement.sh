#!/usr/bin/env bash
# scripts/run_measurement.sh
# Train SGCN embeddings and compute base polarization scores.

set -euo pipefail

cd "$(dirname "$0")/.."

# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------
MODEL="sgcn"
SEEDS=(0)
DATASETS=("bitcoinalpha")

EMB_DIMS=(128)
NUM_LAYERS=(3)
LRS=(0.001)

EPOCHS=400
NEG_SCALE="0.1"
TRAIN_ARGS=""

usage() {
  cat <<EOF
Usage: bash scripts/run_measurement.sh [options]

Options:
  --model sgcn                         Model name. Default: ${MODEL}
  --seeds "0 1 2"                      Seeds. Default: "${SEEDS[*]}"
  --datasets "bitcoinalpha bitcoinotc" Datasets. Default: "${DATASETS[*]}"

  --embedding-dims "128"               Embedding dimensions. Default: "${EMB_DIMS[*]}"
  --num-layers "3"                     Number of SGCN layers. Default: "${NUM_LAYERS[*]}"
  --lrs "0.001"                        Learning rates. Default: "${LRS[*]}"
  --epochs 400                         Number of training epochs. Default: ${EPOCHS}

  --neg-scale 0.1                      Negative edge scale for delta. Default: ${NEG_SCALE}
  --train-args "<args>"                Extra arguments passed to src.train.train.

  -h, --help                           Show this help and exit.

Examples:
  bash scripts/run_measurement.sh

  bash scripts/run_measurement.sh \\
    --datasets "bitcoinalpha bitcoinotc"

  bash scripts/run_measurement.sh \\
    --datasets "wiki-Elec wiki-RfA" \\
    --seeds "0 1 2 3 4"

  bash scripts/run_measurement.sh \\
    --embedding-dims "32 64 128" \\
    --num-layers "2 3 4" \\
    --lrs "0.05 0.01 0.005 0.001 0.0005"
EOF
}

require_value() {
  if [[ $# -lt 2 || -z "${2:-}" ]]; then
    echo "[ERROR] Option $1 requires a value."
    exit 1
  fi
}

# -----------------------------------------------------------------------------
# Parse CLI arguments
# -----------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      require_value "$@"
      MODEL="$2"
      shift 2
      ;;
    --seeds)
      require_value "$@"
      read -r -a SEEDS <<< "$2"
      shift 2
      ;;
    --datasets)
      require_value "$@"
      read -r -a DATASETS <<< "$2"
      shift 2
      ;;
    --embedding-dims|--embedding_dims)
      require_value "$@"
      read -r -a EMB_DIMS <<< "$2"
      shift 2
      ;;
    --num-layers|--num_layers)
      require_value "$@"
      read -r -a NUM_LAYERS <<< "$2"
      shift 2
      ;;
    --lrs)
      require_value "$@"
      read -r -a LRS <<< "$2"
      shift 2
      ;;
    --epochs)
      require_value "$@"
      EPOCHS="$2"
      shift 2
      ;;
    --neg-scale|--neg_scale)
      require_value "$@"
      NEG_SCALE="$2"
      shift 2
      ;;
    --train-args|--train_args)
      require_value "$@"
      TRAIN_ARGS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown option: $1"
      echo ""
      usage
      exit 1
      ;;
  esac
done

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo "============================================================"
echo "[MEASURE] configuration"
echo "  MODEL          = ${MODEL}"
echo "  SEEDS          = ${SEEDS[*]}"
echo "  DATASETS       = ${DATASETS[*]}"
echo "  EMB_DIMS       = ${EMB_DIMS[*]}"
echo "  NUM_LAYERS     = ${NUM_LAYERS[*]}"
echo "  LRS            = ${LRS[*]}"
echo "  EPOCHS         = ${EPOCHS}"
echo "  NEG_SCALE      = ${NEG_SCALE}"
if [[ -n "${TRAIN_ARGS}" ]]; then
  echo "  EXTRA TRAIN_ARGS = ${TRAIN_ARGS}"
else
  echo "  EXTRA TRAIN_ARGS = (none)"
fi
echo "============================================================"
echo ""

# -----------------------------------------------------------------------------
# 1) Train embeddings
# -----------------------------------------------------------------------------
echo "[TRAIN] start"

for seed in "${SEEDS[@]}"; do
  echo ""
  echo "============================================================"
  echo "[TRAIN] seed=${seed}"
  echo "============================================================"

  # shellcheck disable=SC2086
  python -m src.train.train \
    --model "${MODEL}" \
    --seed "${seed}" \
    --datasets "${DATASETS[@]}" \
    --embedding_dims "${EMB_DIMS[@]}" \
    --num_layers "${NUM_LAYERS[@]}" \
    --lrs "${LRS[@]}" \
    --epochs "${EPOCHS}" \
    ${TRAIN_ARGS}
done

echo ""
echo "[TRAIN] finished"
echo ""

# -----------------------------------------------------------------------------
# 2) Base delta measurement
# -----------------------------------------------------------------------------
echo "[DELTA] start"

for seed in "${SEEDS[@]}"; do
  echo ""
  echo "============================================================"
  echo "[DELTA] seed=${seed} neg_scale=${NEG_SCALE}"
  echo "============================================================"

  python -m src.delta.compute_deltas \
    --model "${MODEL}" \
    --seed "${seed}" \
    --neg_scale "${NEG_SCALE}" \
    --datasets "${DATASETS[@]}"
done

echo ""
echo "[SUMMARY] summarizing base results"
python -m src.analysis.summarize_base

python - <<'PY'
from src.utils.paths import RESULTS_BASE

print("")
print("[OUTPUT]")
print(f"  RESULTS_BASE     : {RESULTS_BASE}")
print(f"  base_deltas      : {RESULTS_BASE / 'base_deltas.csv'}")
print(f"  best_embeddings  : {RESULTS_BASE / 'best_embeddings.csv'}")
print(f"  summary_base     : {RESULTS_BASE / 'summary_base.csv'}")
PY

echo ""
echo "[MEASURE] done"
