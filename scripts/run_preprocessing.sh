#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Preprocessing pipeline runner
#
# Usage:
#   bash scripts/run_preprocessing.sh
#   bash scripts/run_preprocessing.sh bitcoinalpha
#   bash scripts/run_preprocessing.sh bitcoinotc
#   bash scripts/run_preprocessing.sh wiki-RfA
#   bash scripts/run_preprocessing.sh wiki-Elec
#   bash scripts/run_preprocessing.sh Slashdot
#   bash scripts/run_preprocessing.sh Epinions
#
# Notes:
# - If no argument is given, the default dataset is "bitcoinalpha".
# - Each command invokes the existing preprocessing modules in order.
# - This script assumes it is executed from the project root.
###############################################################################

# Default dataset
DATASET="${1:-bitcoinalpha}"

echo "[PREP] dataset=${DATASET}"
echo "[PREP] step 1/4: RAW -> INTERIM"
python -m src.preprocessing.process_edge_list "${DATASET}"

echo "[PREP] step 2/4: INTERIM -> BUILD"
python -m src.preprocessing.build_edge_list "${DATASET}"

echo "[PREP] step 3/4: BUILD -> SPLITS"
python -m src.preprocessing.split_graph "${DATASET}"

echo "[PREP] step 4/4: estimate number of communities"
python -m src.preprocessing.estimate_num_communities "${DATASET}"

echo "[PREP] done (${DATASET})"
