#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_preprocessing.sh all
#   bash scripts/run_preprocessing.sh bitcoinalpha
#   bash scripts/run_preprocessing.sh wiki-RfA

TARGET="${1:-all}"
echo "[PREP] target=${TARGET}"

if [[ "${TARGET}" == "all" ]]; then
  python -m src.preprocessing.process_edge_list
  python -m src.preprocessing.build_edge_list
  python -m src.preprocessing.split_graph
  python -m src.preprocessing.estimate_num_communities
else
  python -m src.preprocessing.process_edge_list "${TARGET}"
  python -m src.preprocessing.build_edge_list "${TARGET}"
  python -m src.preprocessing.split_graph "${TARGET}"
  python -m src.preprocessing.estimate_num_communities "${TARGET}"
fi

echo "[PREP] done"
