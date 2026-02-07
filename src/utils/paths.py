# src/utils/paths.py
from __future__ import annotations
from pathlib import Path

# ROOT/src/utils/paths.py -> ROOT is parents[2]
ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
DATA_RAW = DATA / "raw"
DATA_INTERIM = DATA / "interim"
DATA_BUILD = DATA / "build"
DATA_SPLITS = DATA / "splits"
DATA_META = DATA / "meta"

RESULTS = ROOT / "results"
RESULTS_BASE = RESULTS / "base"
RESULTS_GRAY = RESULTS / "gray"
