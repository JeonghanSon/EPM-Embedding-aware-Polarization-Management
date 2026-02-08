#!/usr/bin/env bash
# scripts/run_measurement.sh
# Train + base delta measurement
set -euo pipefail

cd "$(dirname "$0")/.."

# -------------------------
# Defaults (reviewer-friendly)
# -------------------------
SEEDS=(0)
DATASETS=("bitcoinalpha")
NEG_SCALE="0.1"
TRAIN_ARGS=""

usage() {
  cat <<EOF
Usage: bash scripts/run_measurement.sh [options]

Options:
  --seeds "0 1 2"              Seeds (space-separated). Default: "${SEEDS[*]}"
  --datasets "bitcoinalpha"    Datasets (space-separated). Default: "${DATASETS[*]}"
  --neg-scale 0.1              Negative edge scale for delta. Default: ${NEG_SCALE}
  --train-args "<args>"        Extra args passed to src.train.train (optional)

  -h, --help                   Show this help and exit.

Examples:
  bash scripts/run_measurement.sh
  bash scripts/run_measurement.sh --datasets "bitcoinalpha bitcoinotc"
  bash scripts/run_measurement.sh --datasets "Slashdot Epinions" --neg-scale 0.2
  bash scripts/run_measurement.sh --seeds "0" --neg-scale 0.05

  # Grid training (optional)
  bash scripts/run_measurement.sh --train-args "--embedding_dims 32 64 128 --num_layers 2 3 --lrs 0.05 0.01"
EOF
}

# -------------------------
# Parse CLI args
# -------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --seeds)
      read -r -a SEEDS <<< "${2:-}"
      shift 2
      ;;
    --datasets)
      read -r -a DATASETS <<< "${2:-}"
      shift 2
      ;;
    --neg-scale|--neg_scale)
      NEG_SCALE="${2:-}"
      shift 2
      ;;
    --train-args|--train_args)
      TRAIN_ARGS="${2:-}"
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

# -------------------------
# Summary
# -------------------------
echo "============================================================"
echo "Measurement Pipeline"
echo "  SEEDS      = ${SEEDS[*]}"
echo "  DATASETS   = ${DATASETS[*]}"
echo "  NEG_SCALE  = ${NEG_SCALE}"
if [[ -n "${TRAIN_ARGS}" ]]; then
  echo "  TRAIN_ARGS = ${TRAIN_ARGS}"
else
  echo "  TRAIN_ARGS = (default)"
fi
echo "============================================================"
echo ""

# -------------------------
# 1) Train
# -------------------------
echo "🚀 Training start"
for seed in "${SEEDS[@]}"; do
  echo ""
  echo "=============================="
  echo "▶ TRAIN seed=${seed}"
  echo "=============================="

  # shellcheck disable=SC2086
  python -m src.train.train \
    --seed "${seed}" \
    --datasets "${DATASETS[@]}" \
    ${TRAIN_ARGS}
done

echo ""
echo "✅ Training finished"
echo ""

# -------------------------
# 2) Base delta measurement
# -------------------------
echo "📏 Base delta measurement start"
for seed in "${SEEDS[@]}"; do
  echo ""
  echo "=============================="
  echo "▶ DELTA seed=${seed} neg_scale=${NEG_SCALE}"
  echo "=============================="

  python -m src.delta.compute_deltas \
    --seed "${seed}" \
    --neg_scale "${NEG_SCALE}" \
    --datasets "${DATASETS[@]}"
done

echo ""
echo "📌 Summarizing base results"
python -m src.analysis.summarize_base

python - <<'PY'
from src.utils.paths import RESULTS_BASE
print("")
print("📌 Output files")
print(f"  - RESULTS_BASE:  {RESULTS_BASE}")
print(f"  - base_deltas:   {RESULTS_BASE / 'base_deltas.csv'}")
print(f"  - best_embeddings: {RESULTS_BASE / 'best_embeddings.csv'}")
print(f"  - summary_base:  {RESULTS_BASE / 'summary_base.csv'}")
PY
