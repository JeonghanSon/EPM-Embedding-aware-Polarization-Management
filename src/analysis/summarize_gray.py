#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.paths import RESULTS_BASE, RESULTS_GRAY


KEY_COLS = [
    "dataset",
    "gray_mode",
    "scope",
    "pair_selector",
    "minmax",
    "max_degree",
    "topk",
    "setting",
    "edge_scale",
    "neg_scale",
]

AUG_KEY_COLS = KEY_COLS[:-1]  # exclude neg_scale (connect outputs do not depend on neg_scale)

COMPARE_OUT_COLS = [
    "dataset",
    "gray_mode",
    "scope",
    "pair_selector",
    "minmax",
    "max_degree",
    "topk",
    "setting",
    "edge_scale",
    "neg_scale",
    "delta_change",
    "test_f1_change",
]


def _read_csv(path: Path, required: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    return df


def _mean(series: pd.Series) -> float:
    x = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return float(np.nanmean(x)) if np.sum(~np.isnan(x)) > 0 else np.nan


def _std(series: pd.Series) -> float:
    x = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return float(np.nanstd(x, ddof=1)) if np.sum(~np.isnan(x)) >= 2 else np.nan


def _safe_pct_change(new: pd.Series, old: pd.Series) -> pd.Series:
    new = pd.to_numeric(new, errors="coerce")
    old = pd.to_numeric(old, errors="coerce")
    denom = old.replace(0, np.nan)
    return (new - old) / denom


def _dir_selector(pair_selector: str, minmax_th: float, max_degree: float, topk: float | None) -> str:
    mm = f"mm{int(round(float(minmax_th) * 100))}"
    if pd.isna(max_degree):
        deg = "degINF"
    else:
        md = float(max_degree)
        deg = "degINF" if (np.isinf(md) or md >= 1e9) else f"deg{int(md)}"

    if pair_selector == "threshold":
        return f"{pair_selector}_{mm}_{deg}"
    tk = 0 if (topk is None or pd.isna(topk)) else int(topk)
    return f"{pair_selector}_{mm}_{deg}_topk{tk}"


def _dir_escale(edge_scale: float) -> str:
    v = int(round(float(edge_scale) * 100))
    return f"escale_{v//100}p{v%100:02d}"


def run_dir_from_config(row: dict, seed: int) -> Path:
    return (
        RESULTS_GRAY
        / str(row["dataset"])
        / f"gray_scale={row['gray_mode']}"
        / f"scope={row['scope']}"
        / f"pair={_dir_selector(row['pair_selector'], row['minmax'], row['max_degree'], row['topk'])}"
        / f"normalize={row['setting']}"
        / _dir_escale(row["edge_scale"])
        / f"seed{int(seed)}"
    )


def load_augmentation_stats(df_cfg: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df_cfg.iterrows():
        seed = int(r["seed"])
        cfg = {k: r[k] for k in AUG_KEY_COLS}
        run_dir = run_dir_from_config(cfg, seed)
        p = run_dir / "gray_results.csv"

        if not p.exists():
            rows.append({
                **cfg,
                "seed": seed,
                "n_pairs": np.nan,
                "edges_added": np.nan,
                "gg_edges": np.nan,
                "gc_edges": np.nan,
            })
            continue

        gdf = pd.read_csv(p)
        rows.append({
            **cfg,
            "seed": seed,
            "n_pairs": int(len(gdf)),
            "edges_added": float(pd.to_numeric(gdf.get("num_edges_added"), errors="coerce").sum()),
            "gg_edges": float(pd.to_numeric(gdf.get("num_gray_gray_edges"), errors="coerce").sum()),
            "gc_edges": float(pd.to_numeric(gdf.get("num_gray_comm_edges"), errors="coerce").sum()),
        })
    return pd.DataFrame(rows)


def build_summary_gray(deltas_path: Path, best_path: Path) -> pd.DataFrame:
    df_d = _read_csv(deltas_path, KEY_COLS + ["seed", "k", "gray_delta"])
    df_b = _read_csv(best_path, KEY_COLS[:-1] + ["seed", "val_accuracy", "val_f1", "test_accuracy", "test_f1"])

    df_d["seed"] = pd.to_numeric(df_d["seed"], errors="coerce").astype("Int64")
    df_d["k"] = pd.to_numeric(df_d["k"], errors="coerce").astype("Int64")
    df_d["gray_delta"] = pd.to_numeric(df_d["gray_delta"], errors="coerce")
    df_d["neg_scale"] = pd.to_numeric(df_d["neg_scale"], errors="coerce")

    df_b["seed"] = pd.to_numeric(df_b["seed"], errors="coerce").astype("Int64")
    for c in ["val_accuracy", "val_f1", "test_accuracy", "test_f1"]:
        df_b[c] = pd.to_numeric(df_b[c], errors="coerce")

    df = df_d.merge(df_b, on=AUG_KEY_COLS + ["seed"], how="left")

    df_cfg = df[AUG_KEY_COLS + ["seed"]].drop_duplicates().copy()
    df_aug = load_augmentation_stats(df_cfg)
    df = df.merge(df_aug, on=AUG_KEY_COLS + ["seed"], how="left")

    g = df.groupby(KEY_COLS, dropna=False)

    out = g.agg(
        n_seeds_delta=("gray_delta", lambda s: int(s.notna().sum())),
        gray_delta_mean=("gray_delta", _mean),
        gray_delta_std=("gray_delta", _std),

        n_seeds_evaluated=("test_f1", lambda s: int(s.notna().sum())),
        k_mean=("k", _mean),

        val_accuracy_mean=("val_accuracy", _mean),
        val_f1_mean=("val_f1", _mean),

        test_accuracy_mean=("test_accuracy", _mean),
        test_accuracy_std=("test_accuracy", _std),
        test_f1_mean=("test_f1", _mean),
        test_f1_std=("test_f1", _std),

        n_seeds_aug=("n_pairs", lambda s: int(s.notna().sum())),
        n_pairs_mean=("n_pairs", _mean),
        edges_added_mean=("edges_added", _mean),
        gg_edges_mean=("gg_edges", _mean),
        gc_edges_mean=("gc_edges", _mean),
    ).reset_index()

    return out


def build_compare_base_gray(base_summary_path: Path, summary_gray: pd.DataFrame) -> pd.DataFrame:
    base_req = ["dataset", "neg_scale", "base_delta_mean", "test_f1_mean"]
    df_b = _read_csv(base_summary_path, base_req).copy()
    df_b = df_b[base_req].copy()
    df_b["neg_scale"] = pd.to_numeric(df_b["neg_scale"], errors="coerce")
    df_b["base_delta_mean"] = pd.to_numeric(df_b["base_delta_mean"], errors="coerce")
    df_b["base_test_f1_mean"] = pd.to_numeric(df_b["test_f1_mean"], errors="coerce")
    df_b = df_b.drop(columns=["test_f1_mean"])

    df_g = summary_gray.copy()
    df_g["neg_scale"] = pd.to_numeric(df_g["neg_scale"], errors="coerce")
    df_g["gray_delta_mean"] = pd.to_numeric(df_g["gray_delta_mean"], errors="coerce")
    df_g["gray_test_f1_mean"] = pd.to_numeric(df_g["test_f1_mean"], errors="coerce")
    df_g = df_g.drop(columns=["test_f1_mean"])

    df = df_g.merge(df_b, on=["dataset", "neg_scale"], how="left")

    df["delta_change"] = _safe_pct_change(df["gray_delta_mean"], df["base_delta_mean"])
    df["test_f1_change"] = _safe_pct_change(df["gray_test_f1_mean"], df["base_test_f1_mean"])

    return df[COMPARE_OUT_COLS].copy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--deltas", type=str, default=str(RESULTS_GRAY / "gray_deltas.csv"))
    p.add_argument("--best", type=str, default=str(RESULTS_GRAY / "best_embeddings.csv"))
    p.add_argument("--out", type=str, default=str(RESULTS_GRAY / "summary_gray.csv"))

    # Optional: also build comparison vs base
    p.add_argument("--base_summary", type=str, default="", help="Path to summary_base.csv (optional).")
    p.add_argument("--compare_out", type=str, default=str(RESULTS_GRAY / "compare_base_gray.csv"))

    p.add_argument("--print", action="store_true", help="Print summary to stdout.")
    p.add_argument("--top", type=int, default=50, help="Max rows to print when --print is set.")
    args = p.parse_args()

    summary_gray = build_summary_gray(Path(args.deltas), Path(args.best))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_gray.to_csv(out_path, index=False)
    print(f"✅ saved: {out_path}")

    if args.print:
        topn = int(max(0, args.top))
        print(summary_gray.head(topn).to_string(index=False) if topn > 0 else summary_gray.to_string(index=False))

    if str(args.base_summary).strip():
        base_path = Path(args.base_summary)
        compare = build_compare_base_gray(base_path, summary_gray)
        compare_out = Path(args.compare_out)
        compare_out.parent.mkdir(parents=True, exist_ok=True)
        compare.to_csv(compare_out, index=False)
        print(f"✅ saved: {compare_out}")


if __name__ == "__main__":
    main()
