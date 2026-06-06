#!/usr/bin/env bash
# scripts/run_mitigation_prep.sh
# Prepare KMeans communities, PCS pairs, and gray-node scores for EPM mitigation.

set -euo pipefail

cd "$(dirname "$0")/.."

# -------------------------
# Defaults
# -------------------------
SEEDS=(0)
DATASETS=("bitcoinalpha")
NEG_SCALE="0.1"

# Datasets that should use the large-graph PCS pipeline.
LARGE_DATASETS=("Slashdot" "Epinions")

usage() {
  cat <<EOF
Usage: bash scripts/run_mitigation_prep.sh [options]

Options:
  --seeds "0 1 2"              Seeds. Default: "${SEEDS[*]}"
  --datasets "bitcoinalpha"    Datasets. Default: "${DATASETS[*]}"
  --neg-scale 0.1              Negative edge scale for PCS. Default: ${NEG_SCALE}

  -h, --help                   Show this help and exit.

Examples:
  bash scripts/run_mitigation_prep.sh
  bash scripts/run_mitigation_prep.sh --datasets "bitcoinalpha bitcoinotc"
  bash scripts/run_mitigation_prep.sh --datasets "wiki-Elec wiki-RfA" --seeds "0"
  bash scripts/run_mitigation_prep.sh --datasets "Slashdot Epinions"
  bash scripts/run_mitigation_prep.sh --seeds "0 1 2 3 4" --datasets "bitcoinalpha"
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

is_large_dataset() {
  local ds="$1"
  for x in "${LARGE_DATASETS[@]}"; do
    if [[ "$ds" == "$x" ]]; then
      return 0
    fi
  done
  return 1
}

echo "============================================================"
echo "[MITIGATION PREP]"
echo "  SEEDS     = ${SEEDS[*]}"
echo "  DATASETS  = ${DATASETS[*]}"
echo "  NEG_SCALE = ${NEG_SCALE}"
echo "============================================================"
echo ""

for seed in "${SEEDS[@]}"; do
  for ds in "${DATASETS[@]}"; do
    echo "============================================================"
    echo "[KMEANS] dataset=${ds} seed=${seed}"
    echo "============================================================"
    python -m src.gray.kmeans \
      --dataset "${ds}" \
      --seed "${seed}"

    echo "============================================================"
    if is_large_dataset "${ds}"; then
      echo "[PCS_LARGE] dataset=${ds} seed=${seed} neg_scale=${NEG_SCALE}"
      echo "============================================================"
      python -m src.gray.pcs_large \
        --dataset "${ds}" \
        --seed "${seed}" \
        --neg_scale "${NEG_SCALE}"
    else
      echo "[PCS] dataset=${ds} seed=${seed} neg_scale=${NEG_SCALE}"
      echo "============================================================"
      python -m src.gray.pcs \
        --dataset "${ds}" \
        --seed "${seed}" \
        --neg_scale "${NEG_SCALE}"
    fi

    echo "============================================================"
    echo "[GRAY_NODES] dataset=${ds} seed=${seed} normalize=none"
    echo "============================================================"
    python -m src.gray.gray_nodes \
      --dataset "${ds}" \
      --seed "${seed}" \
      --normalize none
  done
done

echo ""
echo "[MITIGATION PREP] finished"
