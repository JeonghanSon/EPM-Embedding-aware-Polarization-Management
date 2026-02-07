#!/usr/bin/env python3
# src/delta/compute_deltas.py
# base delta: best_z.pt -> PCA(k) -> l2 -> delta
# small: build/cached L^dagger -> delta_pinv_multi(L_pinv, X)
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

from src.utils.paths import RESULTS_BASE, DATA_SPLITS, DATA_META
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


def load_best_z(dataset: str, seed: int) -> torch.Tensor:
    p = RESULTS_BASE / dataset / f"seed{seed}" / "best_z.pt"
    z = torch.load(p, map_location="cpu")
    if isinstance(z, dict):
        z = next(iter(z.values()))
    if not isinstance(z, torch.Tensor) or z.ndim != 2:
        raise TypeError(f"{dataset}: best_z.pt must be 2D Tensor")
    return z.detach().cpu().float()


def read_train_edges(dataset: str) -> list[tuple[int, int, float]]:
    p = DATA_SPLITS / dataset / "train_edge_list.csv"
    df = pd.read_csv(p)[["source", "target", "weight"]].dropna()
    return [(int(u), int(v), float(w)) for u, v, w in df.itertuples(index=False)]


def dedup_keep_last_undirected(edge_list: list[tuple[int, int, float]]) -> list[tuple[int, int, float]]:
    """
    Keep-last snapshot by undirected pair (min, max).
    """
    last: dict[tuple[int, int], float] = {}
    for u, v, w in edge_list:
        if w == 0:
            continue
        u = int(u)
        v = int(v)
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        last[(a, b)] = float(w)  # file order = last
    return [(a, b, w) for (a, b), w in last.items()]


def laplacian_pinv_cached(
    dataset: str,
    num_nodes: int,
    edge_list: list[tuple[int, int, float]],
    neg_scale: float,
) -> np.ndarray | None:
    """
    Build L = D - A using sign-scaled weights (+ -> 1.0, - -> neg_scale), then compute pinv(L).
    Cached by (dataset, neg_scale).
    """
    cache_dir = RESULTS_BASE / "laplacian_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    key = f"{dataset}_neg{neg_scale}".replace(".", "p")
    path = cache_dir / f"{key}.npy"
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

    return ((z - mean) @ W).numpy()


def rms_norm(X: np.ndarray) -> np.ndarray:
    rms = np.sqrt((X**2).mean())
    return X / (rms + EPS)


def upsert_csv(path: Path, row: dict):
    cols = ["dataset", "seed", "neg_scale", "k", "base_delta"]
    if path.exists():
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame(columns=cols)

    if not df.empty:
        mask = ~(
            (df["dataset"] == row["dataset"])
            & (df["seed"] == row["seed"])
            & (df["neg_scale"] == row["neg_scale"])
        )
        df = df.loc[mask]

    if df.empty:
        df = pd.DataFrame([row], columns=cols)
    else:
        df = pd.concat([df, pd.DataFrame([row], columns=cols)], ignore_index=True)

    df.to_csv(path, index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--neg_scale", type=float, default=0.1)
    args = p.parse_args()

    seed = int(args.seed)
    neg_scale = float(args.neg_scale)

    if args.datasets:
        datasets = list(args.datasets)
    else:
        datasets = [d.name for d in DATA_SPLITS.iterdir() if d.is_dir()]

    out_csv = RESULTS_BASE / "base_deltas.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    csv_lock = FileLock(str(out_csv) + ".lock")

    for dataset in datasets:
        base_dir = RESULTS_BASE / dataset / f"seed{seed}"
        z_path = base_dir / "best_z.pt"

        if not z_path.exists():
            with csv_lock:
                upsert_csv(
                    out_csv,
                    {"dataset": dataset, "seed": seed, "neg_scale": neg_scale, "k": np.nan, "base_delta": np.nan},
                )
            print(f"[WARN] missing z: {dataset} seed={seed}")
            continue

        try:
            k = load_k(dataset)
            z = load_best_z(dataset, seed)
            X = pca_k(z, k=k, cache_dir=base_dir / "projection_cache")
            X = rms_norm(X)

            edge_list = read_train_edges(dataset)

            if dataset in LARGE_DATASETS:
                delta = delta_solver_multi(edge_list, X, pos=1.0, neg=neg_scale)
            else:
                dedup = dedup_keep_last_undirected(edge_list)
                L_pinv = laplacian_pinv_cached(
                    dataset, num_nodes=X.shape[0], edge_list=dedup, neg_scale=neg_scale
                )
                delta = np.nan if L_pinv is None else delta_pinv_multi(L_pinv, X)

            with csv_lock:
                upsert_csv(
                    out_csv,
                    {"dataset": dataset, "seed": seed, "neg_scale": neg_scale, "k": int(k), "base_delta": float(delta)},
                )
            print(f"base Δ: {dataset} seed={seed} neg={neg_scale} k={k} -> {delta}")

        except Exception as e:
            with csv_lock:
                upsert_csv(
                    out_csv,
                    {"dataset": dataset, "seed": seed, "neg_scale": neg_scale, "k": np.nan, "base_delta": np.nan},
                )
            print(f"[WARN] failed: {dataset} seed={seed} ({e})")

    print(f"saved: {out_csv}")


if __name__ == "__main__":
    main()
