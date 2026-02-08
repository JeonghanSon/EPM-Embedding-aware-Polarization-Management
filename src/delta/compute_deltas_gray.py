#!/usr/bin/env python3
# src/delta/compute_deltas_gray.py
# gray delta: (aug train edges + gray best_z) -> PCA(k) -> norm -> delta
# small: build/cached L^dagger in run_dir -> delta_pinv_multi(L_pinv, X)
# large: delta_solver_multi(edge_list, X)

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from filelock import FileLock

from src.utils.paths import RESULTS_GRAY, DATA_META
from src.delta.delta import delta_pinv_multi, delta_solver_multi

EPS = 1e-12
LARGE_DATASETS = {"Epinions", "Slashdot"}


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


def dedup_keep_last_undirected(edge_list: list[tuple[int, int, float]]) -> list[tuple[int, int, float]]:
    last: dict[tuple[int, int], float] = {}
    for u, v, w in edge_list:
        if w == 0:
            continue
        u = int(u); v = int(v)
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        last[(a, b)] = float(w)  # file order = last
    return [(a, b, w) for (a, b), w in last.items()]


def laplacian_pinv_cached_in_run(
    run_dir: Path,
    num_nodes: int,
    edge_list: list[tuple[int, int, float]],
    neg_scale: float,
) -> np.ndarray | None:
    """
    Build L = D - A using sign-scaled weights (+ -> 1.0, - -> neg_scale), then pinv(L).
    Cached inside run_dir/laplacian_cache by neg_scale.
    """
    cache_dir = run_dir / "laplacian_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    key = f"neg{neg_scale}".replace(".", "p")
    path = cache_dir / f"L_pinv_{key}.npy"
    lock = FileLock(str(path) + ".lock")

    with lock:
        if path.exists():
            try:
                return np.load(path)
            except Exception:
                pass

        try:
            A = np.zeros((num_nodes, num_nodes), dtype=np.float64)
            for u, v, w in edge_list:
                if not (0 <= u < num_nodes and 0 <= v < num_nodes) or u == v:
                    continue
                w_adj = 1.0 if w > 0 else float(neg_scale)
                A[u, v] += w_adj
                A[v, u] += w_adj

            L = np.diag(A.sum(axis=1)) - A
            L_pinv = np.linalg.pinv(L)

            tmp = path.with_suffix(f".tmp.{os.getpid()}")
            with open(tmp, "wb") as f:
                np.save(f, L_pinv)
            os.replace(tmp, path)

            return L_pinv
        except Exception:
            return None


@torch.no_grad()
def pca_k(z: torch.Tensor, k: int, cache_dir: Path) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    N, H = z.shape
    k = int(min(k, H))
    if k <= 0:
        raise ValueError("k must be positive")

    cache_path = cache_dir / f"pca_k{k}_H{H}.pt"
    if cache_path.exists():
        ck = torch.load(cache_path, map_location="cpu")
        mean = ck["mean"]
        W = ck["W"]
    else:
        mean = z.mean(dim=0)
        Zc = z - mean
        _, _, Vt = torch.linalg.svd(Zc, full_matrices=False)
        W = Vt[:k, :].T.contiguous()
        torch.save({"mean": mean, "W": W}, cache_path)

    X = ((z - mean) @ W).numpy()
    rms = np.sqrt((X ** 2).mean())
    return X / (rms + EPS)


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

    dataset = str(args.dataset)
    seed = int(args.seed)
    neg_scale = float(args.neg_scale)

    run_dir = gray_save_dir(
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
        X = pca_k(z, k=k, cache_dir=run_dir / "projection_cache")

        edge_list = read_aug_train_edges(run_dir)

        if dataset in LARGE_DATASETS:
            gray_delta = delta_solver_multi(edge_list, X, pos=1.0, neg=neg_scale)
        else:
            dedup = dedup_keep_last_undirected(edge_list)
            L_pinv = laplacian_pinv_cached_in_run(run_dir, num_nodes=X.shape[0], edge_list=dedup, neg_scale=neg_scale)
            gray_delta = np.nan if L_pinv is None else delta_pinv_multi(L_pinv, X)

        row = dict(row_base)
        row["k"] = int(k)
        row["gray_delta"] = float(gray_delta)

        upsert_csv(out_csv, row, key_cols)
        print(f"✅ gray Δ: {dataset} seed={seed} neg={neg_scale} -> {gray_delta}")
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
