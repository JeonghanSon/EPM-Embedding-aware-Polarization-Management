#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.paths import RESULTS_BASE


def _read_csv(path: Path, required: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    return df


def _nanmean(x: np.ndarray) -> float:
    return float(np.nanmean(x)) if x.size else float("nan")


def _nanstd(x: np.ndarray) -> float:
    ok = np.isfinite(x)
    if int(ok.sum()) >= 2:
        return float(np.nanstd(x, ddof=1))
    return float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--deltas", type=str, default=str(RESULTS_BASE / "base_deltas.csv"))
    p.add_argument("--best", type=str, default=str(RESULTS_BASE / "best_embeddings.csv"))
    p.add_argument("--out", type=str, default=str(RESULTS_BASE / "summary_base.csv"))
    p.add_argument("--top", type=int, default=50)  # number of rows to print
    args = p.parse_args()

    deltas_path = Path(args.deltas)
    best_path = Path(args.best)
    out_path = Path(args.out)

    df_d = _read_csv(deltas_path, ["dataset", "seed", "neg_scale", "k", "base_delta"])
    df_b = _read_csv(
        best_path,
        ["dataset", "seed", "num_communities", "val_accuracy", "val_f1", "test_accuracy", "test_f1"],
    )

    # sanitize types
    df_d["seed"] = pd.to_numeric(df_d["seed"], errors="coerce").astype("Int64")
    df_d["neg_scale"] = pd.to_numeric(df_d["neg_scale"], errors="coerce")
    df_d["k"] = pd.to_numeric(df_d["k"], errors="coerce").astype("Int64")
    df_d["base_delta"] = pd.to_numeric(df_d["base_delta"], errors="coerce")

    df_b["seed"] = pd.to_numeric(df_b["seed"], errors="coerce").astype("Int64")
    df_b["num_communities"] = pd.to_numeric(df_b["num_communities"], errors="coerce").astype("Int64")
    for c in ["val_accuracy", "val_f1", "test_accuracy", "test_f1"]:
        df_b[c] = pd.to_numeric(df_b[c], errors="coerce")

    df_d = df_d.dropna(subset=["dataset", "seed", "neg_scale"]).copy()
    df_b = df_b.dropna(subset=["dataset", "seed"]).copy()

    df_b = df_b[["dataset", "seed", "num_communities", "val_accuracy", "val_f1", "test_accuracy", "test_f1"]]
    df_d = df_d[["dataset", "seed", "neg_scale", "k", "base_delta"]]

    # merge: delta rows (dataset,seed,neg_scale) + perf rows (dataset,seed)
    df = df_d.merge(df_b, on=["dataset", "seed"], how="left")

    g = df.groupby(["dataset", "neg_scale"], dropna=False)

    out = pd.DataFrame({
        "dataset": g["dataset"].first(),
        "neg_scale": g["neg_scale"].first(),

        "n_seeds_delta": g["base_delta"].apply(lambda s: int(pd.notna(s).sum())),
        "base_delta_mean": g["base_delta"].apply(lambda s: _nanmean(s.to_numpy(dtype=float))),
        "base_delta_std": g["base_delta"].apply(lambda s: _nanstd(s.to_numpy(dtype=float))),

        "n_seeds_metrics": g["test_f1"].apply(lambda s: int(pd.notna(s).sum())),
        "num_communities_mean": g["num_communities"].apply(lambda s: _nanmean(s.to_numpy(dtype=float))),
        "val_accuracy_mean": g["val_accuracy"].apply(lambda s: _nanmean(s.to_numpy(dtype=float))),
        "val_f1_mean": g["val_f1"].apply(lambda s: _nanmean(s.to_numpy(dtype=float))),

        "test_accuracy_mean": g["test_accuracy"].apply(lambda s: _nanmean(s.to_numpy(dtype=float))),
        "test_accuracy_std": g["test_accuracy"].apply(lambda s: _nanstd(s.to_numpy(dtype=float))),
        "test_f1_mean": g["test_f1"].apply(lambda s: _nanmean(s.to_numpy(dtype=float))),
        "test_f1_std": g["test_f1"].apply(lambda s: _nanstd(s.to_numpy(dtype=float))),
    }).reset_index(drop=True)

    out = out.sort_values(["dataset", "neg_scale"], ascending=[True, True]).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"✅ saved: {out_path}")
    topn = int(max(0, args.top))
    if topn > 0:
        print(out.head(topn).to_string(index=False))


if __name__ == "__main__":
    main()
