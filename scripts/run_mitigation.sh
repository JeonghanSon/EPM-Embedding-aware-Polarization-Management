#!/usr/bin/env bash
# scripts/run_mitigation.sh
# Mitigation pipeline:
#   1) gray_connect -> augmented train edges
#   2) train_gray   -> best embedding/model on augmented graph
#   3) compute_deltas_gray -> gray delta
#   4) write a simple summary CSV (no auto-selection)
set -euo pipefail

cd "$(dirname "$0")/.."

# -------------------------
# Defaults (reviewer-friendly)
# -------------------------
SEEDS=(0)
DATASETS=("bitcoinalpha")
NEG_SCALE="0.1"
DO_GRID=0

# Large datasets: use fixed params only (no grid recommendation)
LARGE_DATASETS=("Epinions" "Slashdot")

# Fixed mitigation config (not exposed; matches your paper usage)
GRAY_MODE="avg"
SCOPE="global"
PAIR_SELECTOR="threshold"
TOPK=""          # None
SETTING="none"

# Default paper parameters (shown as tau, d_max, gamma)
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
  --neg-scale 0.1              neg_scale used for delta computation. Default: ${NEG_SCALE}
  --grid                       Run recommended grid on non-large datasets.
  -h, --help                   Show this help and exit.

Notes:
  - Large datasets (Epinions/Slashdot) use fixed (tau,d_max,gamma) and ignore --grid.

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
      shift 1
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

  # 1) Build augmented train edges
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

  # 2) Train on augmented edges (grid inside train_gray is kept as-is; we are not changing it here)
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

  # 3) Compute gray delta
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
echo "  MODE      = ${GRAY_MODE}/${SCOPE}/${PAIR_SELECTOR}/${SETTING}"
echo "  GRID      = ${DO_GRID}"
echo "============================================================"
echo ""

# -------------------------
# Run
# -------------------------
for seed in "${SEEDS[@]}"; do
  for ds in "${DATASETS[@]}"; do
    if is_large_dataset "${ds}"; then
      # fixed only
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
        # single representative setting
        run_one "${ds}" "${seed}" "${TAU_SMALL}" "${D_MAX_SMALL}" "${GAMMA_SMALL}"
      fi
    fi
  done
done

# -------------------------
# Lightweight summary (no selection)
# -------------------------
python - <<'PY'
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.utils.paths import RESULTS_GRAY

out_path = RESULTS_GRAY / "mitigation_summary.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)

rows = []

# Collect per-run best metrics + delta + connect summary if present
for dataset_dir in RESULTS_GRAY.iterdir():
    if not dataset_dir.is_dir():
        continue
    dataset = dataset_dir.name
    if dataset in ("laplacian_cache",):  # skip non-dataset folders if any
        continue

    # Walk configs
    for cfg_dir in dataset_dir.glob("gray_scale=*/scope=*/pair=*/normalize=*/escale_*/seed*"):
        if not cfg_dir.is_dir():
            continue

        # parse seed
        try:
            seed = int(cfg_dir.name.replace("seed", ""))
        except Exception:
            seed = None

        best_metrics_path = cfg_dir / "best_metrics.json"
        gray_results_path = cfg_dir / "gray_results.csv"

        # config parts from path
        parts = cfg_dir.parts
        # .../<dataset>/gray_scale=.../scope=.../pair=.../normalize=.../escale_.../seedX
        gray_mode = cfg_dir.parents[4].name.split("=", 1)[1] if len(cfg_dir.parents) >= 5 else None
        scope = cfg_dir.parents[3].name.split("=", 1)[1] if len(cfg_dir.parents) >= 4 else None
        pair = cfg_dir.parents[2].name.split("=", 1)[1] if len(cfg_dir.parents) >= 3 else None
        setting = cfg_dir.parents[1].name.split("=", 1)[1] if len(cfg_dir.parents) >= 2 else None
        escale = cfg_dir.parents[0].name  # escale_...

        val_f1 = test_f1 = val_acc = test_acc = None
        if best_metrics_path.exists():
            try:
                with open(best_metrics_path, "r", encoding="utf-8") as f:
                    m = json.load(f)
                val_f1 = m.get("val_f1", None)
                test_f1 = m.get("test_f1", None)
                val_acc = m.get("val_accuracy", None)
                test_acc = m.get("test_accuracy", None)
            except Exception:
                pass

        selected_pairs = num_edges_added = None
        if gray_results_path.exists():
            try:
                df = pd.read_csv(gray_results_path)
                if not df.empty:
                    selected_pairs = df.get("selected_pairs", None)
                    if selected_pairs is None and "community_1" in df.columns:
                        selected_pairs = int(df["community_1"].nunique() > 0)  # fallback, not used usually
                    if "num_edges_added" in df.columns:
                        num_edges_added = int(df["num_edges_added"].sum())
            except Exception:
                pass

        rows.append({
            "dataset": dataset,
            "seed": seed,
            "gray_mode": gray_mode,
            "scope": scope,
            "pair": pair,
            "setting": setting,
            "escale_dir": escale,
            "val_f1": val_f1,
            "test_f1": test_f1,
            "val_accuracy": val_acc,
            "test_accuracy": test_acc,
            "selected_pairs": selected_pairs,
            "num_edges_added": num_edges_added,
            "run_dir": str(cfg_dir),
        })

# Join with gray_deltas.csv (already aggregated by compute_deltas_gray)
gray_deltas = RESULTS_GRAY / "gray_deltas.csv"
if gray_deltas.exists():
    try:
        ddf = pd.read_csv(gray_deltas)
        # use a wide join key based on config columns present in gray_deltas.csv
        key = ["dataset","seed","gray_mode","scope","pair_selector","minmax","max_degree","topk","setting","edge_scale","neg_scale"]
        # map rows to key columns where possible; keep summary lightweight (no hard failure)
        sdf = pd.DataFrame(rows)
        # if missing columns, just save summary without join
        if all(k in ddf.columns for k in key):
            # parse from run_dir not attempted; just attach gray_deltas separately
            # (users can inspect gray_deltas.csv directly)
            pass
    except Exception:
        pass

pd.DataFrame(rows).to_csv(out_path, index=False)
print(f"✅ saved: {out_path}")
PY

echo ""
echo "✅ Mitigation finished"
