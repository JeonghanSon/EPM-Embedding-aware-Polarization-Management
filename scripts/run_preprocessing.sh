#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Preprocessing pipeline runner
#
# Usage:
#   bash scripts/run_preprocessing.sh
#   bash scripts/run_preprocessing.sh bitcoinalpha
#   bash scripts/run_preprocessing.sh bitcoinotc wiki-Elec wiki-RfA
#   bash scripts/run_preprocessing.sh --dataset Slashdot
#
# Notes:
# - If no dataset is given, this script runs the default dataset: bitcoinalpha.
# - Raw files under data/raw are never deleted.
# - Existing preprocessing outputs for the selected dataset are removed before
#   regeneration.
###############################################################################

cd "$(dirname "$0")/.."

DEFAULT_DATASETS=("bitcoinalpha")

if [[ $# -eq 0 ]]; then
  DATASETS=("${DEFAULT_DATASETS[@]}")
elif [[ "$1" == "--dataset" || "$1" == "-d" ]]; then
  if [[ $# -lt 2 ]]; then
    echo "Error: --dataset requires a dataset name."
    exit 1
  fi
  DATASETS=("${@:2}")
else
  DATASETS=("$@")
fi

echo "============================================================"
echo "[PREP] datasets=${DATASETS[*]}"
echo "============================================================"

for DATASET in "${DATASETS[@]}"; do
  echo ""
  echo "============================================================"
  echo "[PREP] dataset=${DATASET}"
  echo "============================================================"

  echo "[PREP] clean previous preprocessing outputs"
  rm -rf "data/interim/${DATASET}"
  rm -rf "data/build/${DATASET}"
  rm -rf "data/splits/${DATASET}"
  rm -rf "data/meta/${DATASET}"

  echo "[PREP] step 1/4: RAW -> INTERIM"
  python -m src.preprocessing.process_edge_list "${DATASET}"

  echo "[PREP] step 2/4: INTERIM -> BUILD"
  python -m src.preprocessing.build_edge_list "${DATASET}"

  echo "[PREP] step 3/4: BUILD -> SPLITS"
  python -m src.preprocessing.split_graph "${DATASET}"

  echo "[PREP] step 4/4: estimate number of communities"
  python -m src.preprocessing.estimate_num_communities "${DATASET}"

  echo "[PREP] done (${DATASET})"
done

echo ""
echo "[PREP] preprocessing finished: ${DATASETS[*]}"
