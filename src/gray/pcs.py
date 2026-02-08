#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from filelock import FileLock

from src.utils.paths import RESULTS_BASE, RESULTS_GRAY, DATA_META, DATA_SPLITS
from src.delta.delta import delta_solver_multi, delta_pinv_multi  

MIN_COMM_SIZE = 30
LARGE_DATASETS = {"Epinions", "Slashdot"}
EPS = 1e-12


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


def load_kmeans_communities(dataset: str, seed: int) -> dict[int, list[int]]:
    p = RESULTS_GRAY / dataset / f"seed{seed}" / "aug" / "kmeans_community.csv"
    df = pd.read_csv(p)
    out = {}
    for _, r in df.iterrows():
        cid = int(r["community_id"])
        try:
            nodes = eval(r["nodes"])
        except Exception:
            nodes = []
        nodes = [int(x) for x in nodes]
        if len(nodes) >= MIN_COMM_SIZE:
            out[cid] = nodes
    return out


def pca_k(z: torch.Tensor, k: int, cache_dir: Path) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    N, H = z.shape
    k = min(int(k), H)
    cache = cache_dir / f"pca_k{k}_H{H}.pt"
    if cache.exists():
        ck = torch.load(cache, map_location="cpu")
        mean, W = ck["mean"], ck["W"]
    else:
        mean = z.mean(dim=0)
        Zc = z - mean
        _, _, Vt = torch.linalg.svd(Zc, full_matrices=False)
        W = Vt[:k, :].T.contiguous()
        torch.save({"mean": mean, "W": W}, cache)
    X = ((z - mean) @ W).numpy()
    rms = np.sqrt((X ** 2).mean())
    #n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (rms + EPS)


def dedup_keep_last_undirected(edge_list: list[tuple[int, int, float]]) -> list[tuple[int, int, float]]:
    last = {}
    for u, v, w in edge_list:
        if w == 0:
            continue
        u = int(u); v = int(v)
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        last[(a, b)] = float(w)
    return [(a, b, w) for (a, b), w in last.items()]


def load_or_build_pinv(dataset: str, num_nodes: int, edge_list: list[tuple[int, int, float]], neg_scale: float) -> np.ndarray | None:
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




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--neg_scale", type=float, default=0.1)
    args = ap.parse_args()

    dataset, seed, neg = args.dataset, int(args.seed), float(args.neg_scale)

    k = load_k(dataset)
    z = load_best_z(dataset, seed)
    X = pca_k(z, k, cache_dir=RESULTS_GRAY / dataset / f"seed{seed}" / "aug" / "pca_cache")

    edge_list = read_train_edges(dataset)
    comms = load_kmeans_communities(dataset, seed)
    ids = sorted(comms.keys())

    L_pinv = None
    if dataset not in LARGE_DATASETS:
        dedup = dedup_keep_last_undirected(edge_list)
        L_pinv = load_or_build_pinv(dataset, num_nodes=X.shape[0], edge_list=dedup, neg_scale=neg)
        if L_pinv is None:
            print(f"[WARN] pinv unavailable for {dataset} neg={neg} -> deltas will be NaN")

    rows = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            c1, c2 = ids[i], ids[j]
            nodes = comms[c1] + comms[c2]
            Xc = X[nodes]
            Xc = Xc - Xc.mean(axis=0, keepdims=True)
            masked = np.zeros_like(X)
            masked[nodes] = Xc

            if dataset in LARGE_DATASETS:
                d = delta_solver_multi(edge_list, masked, pos=1.0, neg=neg)
            else:
                d = np.nan if L_pinv is None else delta_pinv_multi(L_pinv, masked)  

            rows.append({
                "dataset": dataset,
                "seed": seed,
                "neg_scale": neg,
                "community_1": c1,
                "community_2": c2,
                "size_c1": len(comms[c1]),
                "size_c2": len(comms[c2]),
                "delta": float(d),
            })

    out_dir = RESULTS_GRAY / dataset / f"seed{seed}" / "aug" / "pcs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "delta_pairs.csv"
    pd.DataFrame(rows).sort_values("delta", ascending=False).to_csv(out_csv, index=False)
    print(f"✅ PCS saved: {out_csv} (rows={len(rows)})")


if __name__ == "__main__":
    main()
