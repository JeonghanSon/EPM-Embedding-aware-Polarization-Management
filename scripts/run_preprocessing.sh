#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_preprocessing.sh all
#   bash scripts/run_preprocessing.sh bitcoinalpha
#
# Assumes you run this at the project root:
#   cd <REPO_ROOT>

TARGET="${1:-all}"

echo "[PREP] target=${TARGET}"

run_all () {
  python -m src.preprocessing.process_edge_list
  python -m src.preprocessing.build_edge_list
  python -m src.preprocessing.split_graph
  python src/preprocessing_estimate_num_communities.py
}

run_one () {
  local D="$1"
  echo "[PREP] dataset=${D}"

  # 1) RAW -> INTERIM (dataset-specific)
  python -m src.preprocessing.process_edge_list "${D}"

  # 2) INTERIM -> BUILD
  python -m src.preprocessing.build_edge_list "${D}"

  # 3) BUILD -> SPLITS
  python -m src.preprocessing.split_graph "${D}"

  # 4) Estimate num communities (uses train split)
  python src/preprocessing_estimate_num_communities.py "${D}"
}

if [[ "${TARGET}" == "all" ]]; then
  run_all
else
  run_one "${TARGET}"
fi

echo "[PREP] done"
