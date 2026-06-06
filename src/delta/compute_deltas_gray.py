#!/usr/bin/env python3
# src/delta/compute_deltas_gray.py
# gray delta: (aug train edges + gray best_z) -> PCA(k) -> node-wise L2 -> delta (solver-only)

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from filelock import FileLock

from src.utils.paths import RESULTS_GRAY, DATA_META
from src.delta.delta import delta_solver_multi

EPS = 1e-12


def load_k(dataset: str) -> int:
    p = DATA_META / dataset / "num_communities.json"
    with open(p, "r", encoding="utf-8") as f:
        meta = json.load(f)
    k = meta.get("mode", meta.get("avg", None))
    if k is None:
        raise RuntimeError(f"{dataset}: num_communities.json missing mode/avg")
    return int(k)


def _dir_selector(pair_selector: str, minmax_th: float, max_degree: float, topk: int | None) -> str:
    mm = f"mm{int(round(minmax_th * 100))}"
    deg = "degINF" if (np.isinf(max_degree) or max_degree >= 1e9) else f"deg{int(max_degree)}"
    if pair_selector == "threshold":
        return f"{pair_selector}_{mm}_{deg}"
    return f"{pair_selector}_{mm}_{deg}_topk{int(topk or 0)}"


def _dir_escale(edge_scale: float) -> str:
    v = int(round(edge_scale * 100))
    return f"escale_{v//100}p{v%100:02d}"


def gray_save_dir(
    model: str,
    dataset: str,
    seed: int,
    gray_mode: str,
    scope: str,
    pair_selector: str,
    minmax_th: float,
    max_degree: float,
    topk: int | None,
    setting: str,
    edge_scale: float,
) -> Path:
    return (
        RESULTS_GRAY
        / model
        / dataset
        / f"gray_scale={gray_mode}"
        / f"scope={scope}"
        / f"pair={_dir_selector(pair_selector, minmax_th, max_degree, topk)}"
        / f"normalize={setting}"
        / _dir_escale(edge_scale)
        / f"seed{seed}"
    )


def load_best_z(run_dir: Path) -> torch.Tensor:
    p = run_dir / "best_z.pt"
    z = torch.load(p, map_location="cpu")
    if isinstance(z, dict):
        z = next(iter(z.values()))
    if not isinstance(z, torch.Tensor) or z.ndim != 2:
        raise TypeError(f"{p}: best_z.pt must be 2D Tensor")
    return z.detach().cpu().float()


def read_aug_train_edges(run_dir: Path) -> list[tuple[int, int, float]]:
    p = run_dir / "train_edge_list_aug.csv"
    df = pd.read_csv(p)[["source", "target", "weight"]].dropna()
    return [(int(u), int(v), float(w)) for u, v, w in df.itertuples(index=False)]


@torch.no_grad()
def pca_k(z: torch.Tensor, k: int) -> np.ndarray:
    _, h = z.shape
    k = int(min(k, h))
    if k <= 0:
        raise ValueError("k must be positive")

    mean = z.mean(dim=0)
    Zc = z - mean
    _, _, Vt = torch.linalg.svd(Zc, full_matrices=False)
    W = Vt[:k, :].T.contiguous()

    X = ((z - mean) @ W).numpy()
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (n + EPS)


def upsert_csv(path: Path, row: dict, key_cols: list[str]):
    lock = FileLock(str(path) + ".lock")
    with lock:
        if path.exists():
            df = pd.read_csv(path)
        else:
            df = pd.DataFrame(columns=list(row.keys()))

        for c in row.keys():
            if c not in df.columns:
                df[c] = np.nan

        mask = np.ones(len(df), dtype=bool)
        for kc in key_cols:
            mask &= (df[kc].astype(str) == str(row[kc]))
        df = df.loc[~mask].copy()

        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="sgcn")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, required=True)

    ap.add_argument("--gray_mode", choices=["min", "avg", "max"], required=True)
    ap.add_argument("--scope", choices=["global", "local"], required=True)

    ap.add_argument("--pair_selector", choices=["threshold", "topk"], required=True)
    ap.add_argument("--minmax", type=float, required=True)
    ap.add_argument("--max_degree", type=float, required=True)
    ap.add_argument("--topk", type=int, default=None)

    ap.add_argument("--setting", choices=["none", "l2"], required=True)
    ap.add_argument("--edge_scale", type=float, required=True)
    ap.add_argument("--neg_scale", type=float, default=0.1)
    args = ap.parse_args()

    model = str(args.model)
    dataset = str(args.dataset)
    seed = int(args.seed)
    neg_scale = float(args.neg_scale)

    run_dir = gray_save_dir(
        model=model,
        dataset=dataset,
        seed=seed,
        gray_mode=args.gray_mode,
        scope=args.scope,
        pair_selector=args.pair_selector,
        minmax_th=float(args.minmax),
        max_degree=float(args.max_degree),
        topk=args.topk,
        setting=str(args.setting),
        edge_scale=float(args.edge_scale),
    )

    out_csv = RESULTS_GRAY / "gray_deltas.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    row_base = {
        "model": model,
        "dataset": dataset,
        "seed": seed,
        "gray_mode": args.gray_mode,
        "scope": args.scope,
        "pair_selector": args.pair_selector,
        "minmax": float(args.minmax),
        "max_degree": float(args.max_degree),
        "topk": int(args.topk) if args.topk is not None else None,
        "setting": str(args.setting),
        "edge_scale": float(args.edge_scale),
        "neg_scale": neg_scale,
    }
    key_cols = list(row_base.keys())

    try:
        if not (run_dir / "train_edge_list_aug.csv").exists():
            raise FileNotFoundError(run_dir / "train_edge_list_aug.csv")
        if not (run_dir / "best_z.pt").exists():
            raise FileNotFoundError(run_dir / "best_z.pt")

        k = load_k(dataset)
        z = load_best_z(run_dir)
        X = pca_k(z, k=k)

        edge_list = read_aug_train_edges(run_dir)
        gray_delta = delta_solver_multi(edge_list, X, pos=1.0, neg=neg_scale)

        row = dict(row_base)
        row["k"] = int(k)
        row["gray_delta"] = float(gray_delta)

        upsert_csv(out_csv, row, key_cols)
        print(f"✅ gray Δ: model={model} dataset={dataset} seed={seed} neg={neg_scale} -> {gray_delta}")
        return

    except Exception as e:
        row = dict(row_base)
        row["k"] = np.nan
        row["gray_delta"] = np.nan
        upsert_csv(out_csv, row, key_cols)
        print(f"[WARN] failed gray Δ: {run_dir} ({e})")
        return


if __name__ == "__main__":
    main()
