#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.utils.paths import RESULTS_BASE, RESULTS_GRAY, DATA_META, DATA_SPLITS
from src.delta.delta import delta_solver_multi

MIN_COMM_SIZE = 30
EPS = 1e-12


def load_k(dataset: str) -> int:
    p = DATA_META / dataset / "num_communities.json"
    with open(p, "r", encoding="utf-8") as f:
        meta = json.load(f)
    k = meta.get("mode", meta.get("avg", None))
    if k is None:
        raise RuntimeError(f"{dataset}: num_communities.json missing mode/avg")
    return int(k)


def load_best_z(model: str, dataset: str, seed: int) -> torch.Tensor:
    p = RESULTS_BASE / model / dataset / f"seed{seed}" / "best_z.pt"
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


def load_kmeans_communities(model: str, dataset: str, seed: int) -> dict[int, list[int]]:
    p = RESULTS_GRAY / model / dataset / f"seed{seed}" / "aug" / "kmeans_community.csv"
    df = pd.read_csv(p)

    out: dict[int, list[int]] = {}
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

    _, h = z.shape
    k = min(int(k), h)
    cache = cache_dir / f"pca_k{k}_H{h}.pt"

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
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (n + EPS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="sgcn")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--neg_scale", type=float, default=0.1)
    args = ap.parse_args()

    model = str(args.model)
    dataset = str(args.dataset)
    seed = int(args.seed)
    neg = float(args.neg_scale)

    k = load_k(dataset)
    z = load_best_z(model, dataset, seed)

    aug_dir = RESULTS_GRAY / model / dataset / f"seed{seed}" / "aug"
    X = pca_k(z, k, cache_dir=aug_dir / "pca_cache")

    edge_list = read_train_edges(dataset)
    comms = load_kmeans_communities(model, dataset, seed)
    ids = sorted(comms.keys())

    rows = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            c1, c2 = ids[i], ids[j]
            nodes = comms[c1] + comms[c2]

            Xc = X[nodes]
            Xc = Xc - Xc.mean(axis=0, keepdims=True)

            masked = np.zeros_like(X)
            masked[nodes] = Xc

            d = delta_solver_multi(edge_list, masked, pos=1.0, neg=neg)

            rows.append({
                "model": model,
                "dataset": dataset,
                "seed": seed,
                "neg_scale": neg,
                "community_1": c1,
                "community_2": c2,
                "size_c1": len(comms[c1]),
                "size_c2": len(comms[c2]),
                "delta": float(d),
            })

    out_dir = aug_dir / "pcs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "delta_pairs.csv"

    if rows:
        pd.DataFrame(rows).sort_values("delta", ascending=False).to_csv(out_csv, index=False)
    else:
        pd.DataFrame(columns=[
            "model", "dataset", "seed", "neg_scale",
            "community_1", "community_2", "size_c1", "size_c2", "delta",
        ]).to_csv(out_csv, index=False)

    print(f"✅ PCS saved: model={model} dataset={dataset} seed={seed} -> {out_csv} (rows={len(rows)})")


if __name__ == "__main__":
    main()
