#!/usr/bin/env bash
# scripts/run_mitigation.sh
# Mitigation pipeline (after mitigation prep):
#   1) gray_connect
#   2) train_gray
#   3) compute_deltas_gray
#   4) summarize_gray (+ compare vs base)
set -euo pipefail

cd "$(dirname "$0")/.."

# -------------------------
# Defaults (reviewer-friendly)
# -------------------------
SEEDS=(0)
DATASETS=("bitcoinalpha")
NEG_SCALE="0.1"
DO_GRID=0

# Large datasets: fixed params only (no grid)
LARGE_DATASETS=("Epinions" "Slashdot")

# Fixed mitigation config (implicit; not exposed)
GRAY_MODE="avg"
SCOPE="global"
PAIR_SELECTOR="threshold"
SETTING="none"

# Paper parameters (notation: tau, d_max, gamma)
TAU_SMALL="0.7"
D_MAX_SMALL="3"
GAMMA_SMALL="1.0"

TAU_LARGE="0.2"
D_MAX_LARGE="4"
GAMMA_LARGE="1.0"

# Recommended grid (small/medium datasets only)
GRID_TAUS=("0.7" "0.8" "0.9")
GRID_DMAX=("2" "3")
GRID_GAMMA=("0.5" "1.0" "1.5")

usage() {
  cat <<EOF
Usage: bash scripts/run_mitigation.sh [options]

Options:
  --seeds "0 1 2"              Seeds (space-separated). Default: "${SEEDS[*]}"
  --datasets "bitcoinalpha"    Datasets (space-separated). Default: "${DATASETS[*]}"
  --neg-scale 0.1              neg_scale for delta computation. Default: ${NEG_SCALE}
  --grid                       Run recommended grid (non-large datasets only)
  -h, --help                   Show this help and exit.

Notes:
  - Run mitigation preparation first:
      bash scripts/run_mitigation_prep.sh
  - Large datasets (Epinions/Slashdot) always use fixed parameters.

Examples:
  bash scripts/run_mitigation.sh
  bash scripts/run_mitigation.sh --datasets "bitcoinalpha bitcoinotc"
  bash scripts/run_mitigation.sh --datasets "wiki-Elec" --grid
EOF
}

# -------------------------
# Parse CLI
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
    --grid)
      DO_GRID=1
      shift
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

run_one() {
  local ds="$1"
  local seed="$2"
  local tau="$3"
  local dmax="$4"
  local gamma="$5"

  echo "============================================================"
  echo "[MITIGATION] dataset=${ds} seed=${seed}"
  echo "  (paper params) tau=${tau} d_max=${dmax} gamma=${gamma} | neg_scale=${NEG_SCALE}"
  echo "============================================================"

  python -m src.gray.gray_connect \
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
    --dataset "${ds}" \
    --seed "${seed}" \
    --gray_mode "${GRAY_MODE}" \
    --scope "${SCOPE}" \
    --pair_selector "${PAIR_SELECTOR}" \
    --minmax "${tau}" \
    --max_degree "${dmax}" \
    --setting "${SETTING}" \
    --edge_scale "${gamma}"

  python -m src.delta.compute_deltas_gray \
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
echo "Mitigation Pipeline"
echo "  SEEDS     = ${SEEDS[*]}"
echo "  DATASETS  = ${DATASETS[*]}"
echo "  NEG_SCALE = ${NEG_SCALE}"
echo "  GRID      = ${DO_GRID}"
echo "============================================================"
echo ""

# -------------------------
# Run
# -------------------------
for seed in "${SEEDS[@]}"; do
  for ds in "${DATASETS[@]}"; do
    if is_large_dataset "${ds}"; then
      run_one "${ds}" "${seed}" "${TAU_LARGE}" "${D_MAX_LARGE}" "${GAMMA_LARGE}"
    else
      if [[ "${DO_GRID}" -eq 1 ]]; then
        for tau in "${GRID_TAUS[@]}"; do
          for dmax in "${GRID_DMAX[@]}"; do
            for gamma in "${GRID_GAMMA[@]}"; do
              run_one "${ds}" "${seed}" "${tau}" "${dmax}" "${gamma}"
            done
          done
        done
      else
        run_one "${ds}" "${seed}" "${TAU_SMALL}" "${D_MAX_SMALL}" "${GAMMA_SMALL}"
      fi
    fi
  done
done

echo ""
echo "📌 Summarizing gray results (+ compare vs base)"
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
print("📌 Output files")
print(f"  - RESULTS_GRAY:        {RESULTS_GRAY}")
print(f"  - gray_deltas:         {RESULTS_GRAY / 'gray_deltas.csv'}")
print(f"  - best_embeddings:     {RESULTS_GRAY / 'best_embeddings.csv'}")
print(f"  - summary_gray:        {RESULTS_GRAY / 'summary_gray.csv'}")
print(f"  - compare_base_gray:   {RESULTS_GRAY / 'compare_base_gray.csv'}")
print("")
print("📌 Base summary (used for comparison)")
print(f"  - summary_base:        {RESULTS_BASE / 'summary_base.csv'}")
PY

echo ""
echo "✅ Mitigation finished"
