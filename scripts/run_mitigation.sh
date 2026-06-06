#!/usr/bin/env bash
# scripts/run_mitigation.sh
# Run EPM mitigation after mitigation preparation.

set -euo pipefail

cd "$(dirname "$0")/.."

# -------------------------
# Defaults
# -------------------------
MODEL="sgcn"
SEEDS=(0)
DATASETS=("bitcoinalpha")
NEG_SCALE="0.1"

# Reviewer-friendly fixed mitigation setting.
TAUS=(0.5)
D_MAXS=(3)
GAMMAS=(1.5)

# Reviewer-friendly fixed SGCN setting for gray training.
EMB_DIMS=(128)
LAYERS=(4)
LRS=(0.001)

EPOCHS=400
PATIENCE=10

# Fixed EPM configuration.
GRAY_MODE="avg"
SCOPE="global"
PAIR_SELECTOR="threshold"
SETTING="none"

usage() {
  cat <<EOF
Usage: bash scripts/run_mitigation.sh [options]

Options:
  --seeds "0 1 2"                  Seeds. Default: "${SEEDS[*]}"
  --datasets "bitcoinalpha"        Datasets. Default: "${DATASETS[*]}"
  --neg-scale 0.1                  Negative edge scale for delta. Default: ${NEG_SCALE}

  --tau "0.5 0.6 0.7"              Tau values. Default: "${TAUS[*]}"
  --d-max "2 3"                    d_max values. Default: "${D_MAXS[*]}"
  --gamma "0.5 1.0 1.5"            Gamma values. Default: "${GAMMAS[*]}"

  --embedding-dims "32 64 128"     Embedding dimensions. Default: "${EMB_DIMS[*]}"
  --layers "2 3 4"                 Number of SGCN layers. Default: "${LAYERS[*]}"
  --lrs "0.05 0.01 0.001"          Learning rates. Default: "${LRS[*]}"
  --epochs 400                     Training epochs. Default: ${EPOCHS}
  --patience 10                    Early stopping patience. Default: ${PATIENCE}

  -h, --help                       Show this help and exit.

Examples:
  bash scripts/run_mitigation.sh

  bash scripts/run_mitigation.sh \\
    --datasets "bitcoinalpha bitcoinotc"

  bash scripts/run_mitigation.sh \\
    --seeds "0 1 2 3 4" \\
    --datasets "bitcoinalpha"

  bash scripts/run_mitigation.sh \\
    --tau "0.5 0.6 0.7 0.8 0.9" \\
    --d-max "2 3" \\
    --gamma "0.5 1.0 1.5" \\
    --embedding-dims "32 64 128" \\
    --layers "2 3 4" \\
    --lrs "0.05 0.01 0.005 0.001 0.0005"
EOF
}

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
    --tau)
      read -r -a TAUS <<< "${2:-}"
      shift 2
      ;;
    --d-max|--d_max)
      read -r -a D_MAXS <<< "${2:-}"
      shift 2
      ;;
    --gamma)
      read -r -a GAMMAS <<< "${2:-}"
      shift 2
      ;;
    --embedding-dims|--embedding_dims)
      read -r -a EMB_DIMS <<< "${2:-}"
      shift 2
      ;;
    --layers|--num-layers|--num_layers)
      read -r -a LAYERS <<< "${2:-}"
      shift 2
      ;;
    --lrs)
      read -r -a LRS <<< "${2:-}"
      shift 2
      ;;
    --epochs)
      EPOCHS="${2:-}"
      shift 2
      ;;
    --patience)
      PATIENCE="${2:-}"
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

run_one() {
  local ds="$1"
  local seed="$2"
  local tau="$3"
  local dmax="$4"
  local gamma="$5"

  echo "============================================================"
  echo "[MITIGATION] dataset=${ds} seed=${seed}"
  echo "  tau=${tau} d_max=${dmax} gamma=${gamma} neg_scale=${NEG_SCALE}"
  echo "============================================================"

  python -m src.gray.gray_connect \
    --model "${MODEL}" \
    --dataset "${ds}" \
    --seed "${seed}" \
    --gray_mode "${GRAY_MODE}" \
    --scope "${SCOPE}" \
    --pair_selector "${PAIR_SELECTOR}" \
    --minmax "${tau}" \
    --max_degree "${dmax}" \
    --setting "${SETTING}" \
    --edge_scale "${gamma}"

  python -m src.train.train_gray \
    --model "${MODEL}" \
    --dataset "${ds}" \
    --seed "${seed}" \
    --gray_mode "${GRAY_MODE}" \
    --scope "${SCOPE}" \
    --pair_selector "${PAIR_SELECTOR}" \
    --minmax "${tau}" \
    --max_degree "${dmax}" \
    --setting "${SETTING}" \
    --edge_scale "${gamma}" \
    --embedding_dims "${EMB_DIMS[@]}" \
    --num_layers "${LAYERS[@]}" \
    --lrs "${LRS[@]}" \
    --epochs "${EPOCHS}" \
    --patience "${PATIENCE}"

  python -m src.delta.compute_deltas_gray \
    --model "${MODEL}" \
    --dataset "${ds}" \
    --seed "${seed}" \
    --gray_mode "${GRAY_MODE}" \
    --scope "${SCOPE}" \
    --pair_selector "${PAIR_SELECTOR}" \
    --minmax "${tau}" \
    --max_degree "${dmax}" \
    --setting "${SETTING}" \
    --edge_scale "${gamma}" \
    --neg_scale "${NEG_SCALE}"
}

echo "============================================================"
echo "[MITIGATION]"
echo "  MODEL     = ${MODEL}"
echo "  SEEDS     = ${SEEDS[*]}"
echo "  DATASETS  = ${DATASETS[*]}"
echo "  NEG_SCALE = ${NEG_SCALE}"
echo "  TAUS      = ${TAUS[*]}"
echo "  D_MAXS    = ${D_MAXS[*]}"
echo "  GAMMAS    = ${GAMMAS[*]}"
echo "  GRID      = dim(${EMB_DIMS[*]}) layers(${LAYERS[*]}) lr(${LRS[*]})"
echo "  EPOCHS    = ${EPOCHS}"
echo "  PATIENCE  = ${PATIENCE}"
echo "============================================================"
echo ""

for seed in "${SEEDS[@]}"; do
  for ds in "${DATASETS[@]}"; do
    for tau in "${TAUS[@]}"; do
      for dmax in "${D_MAXS[@]}"; do
        for gamma in "${GAMMAS[@]}"; do
          run_one "${ds}" "${seed}" "${tau}" "${dmax}" "${gamma}"
        done
      done
    done
  done
done

echo ""
echo "[SUMMARY] summarizing gray results and comparing with base"
python -m src.analysis.summarize_gray \
  --base_summary "$(python - <<'PY'
from src.utils.paths import RESULTS_BASE
print(RESULTS_BASE / "summary_base.csv")
PY
)" \
  --print

python - <<'PY'
from src.utils.paths import RESULTS_BASE, RESULTS_GRAY

print("")
print("[OUTPUT]")
print(f"  RESULTS_GRAY       : {RESULTS_GRAY}")
print(f"  gray_deltas        : {RESULTS_GRAY / 'gray_deltas.csv'}")
print(f"  best_embeddings    : {RESULTS_GRAY / 'best_embeddings.csv'}")
print(f"  summary_gray       : {RESULTS_GRAY / 'summary_gray.csv'}")
print(f"  compare_base_gray  : {RESULTS_GRAY / 'compare_base_gray.csv'}")
print("")
print("[BASE]")
print(f"  summary_base       : {RESULTS_BASE / 'summary_base.csv'}")
PY

echo ""
echo "[MITIGATION] finished"
