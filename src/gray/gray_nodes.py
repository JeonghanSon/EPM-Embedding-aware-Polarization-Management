#!/usr/bin/env python3
# Gray node scoring in opinion space (PCA k-dim)
# balance + proximity based gray score
from __future__ import annotations

import argparse
import json
import ast
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch

from src.utils.paths import RESULTS_BASE, RESULTS_GRAY, DATA_META

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


def load_kmeans_communities(model: str, dataset: str, seed: int) -> dict[int, list[int]]:
    p = RESULTS_GRAY / model / dataset / f"seed{seed}" / "aug" / "kmeans_community.csv"
    df = pd.read_csv(p)

    comms: dict[int, list[int]] = {}
    for _, r in df.iterrows():
        cid = int(r["community_id"])
        try:
            nodes = ast.literal_eval(str(r["nodes"]))
        except Exception:
            nodes = []
        nodes = [int(x) for x in nodes]
        if len(nodes) >= MIN_COMM_SIZE:
            comms[cid] = nodes
    return comms


@torch.no_grad()
def pca_k(z: torch.Tensor, k: int, cache_dir: Path) -> np.ndarray:
    """
    PCA projection to k dims (raw).
    Cache mean/W per (model, dataset, seed, k, H) via cache_dir.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    _, h = z.shape
    k = min(int(k), h)

    cache_path = cache_dir / f"pca_k{k}_H{h}.pt"
    if cache_path.exists():
        ck = torch.load(cache_path, map_location="cpu")
        mean, W = ck["mean"], ck["W"]
    else:
        mean = z.mean(dim=0)
        Zc = z - mean
        _, _, Vt = torch.linalg.svd(Zc, full_matrices=False)
        W = Vt[:k, :].T.contiguous()
        torch.save({"mean": mean, "W": W}, cache_path)

    X = ((z - mean) @ W).numpy()  # [N, k]
    return X


def l2_normalize(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (n + EPS)


def compute_gray_scores(
    X: np.ndarray,
    C1: List[int],
    C2: List[int],
    alpha: float = 1.0,
    beta: float = 1.0,
) -> List[Tuple[int, float]]:
    """
    score(i) = alpha * |d1 - d2| + beta * max(d1, d2)
    d*: Euclidean distance to community center.
    Lower score => more gray.
    """
    C1 = list(C1)
    C2 = list(C2)

    c1_center = X[C1].mean(axis=0)
    c2_center = X[C2].mean(axis=0)

    excluded = set(C1).union(C2)
    scores: List[Tuple[int, float]] = []

    for i in range(len(X)):
        if i in excluded:
            continue

        x = X[i]
        d1 = float(np.linalg.norm(x - c1_center))
        d2 = float(np.linalg.norm(x - c2_center))
        score = alpha * abs(d1 - d2) + beta * max(d1, d2)
        scores.append((i, score))

    return sorted(scores, key=lambda t: t[1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="sgcn")
    p.add_argument("--dataset", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--normalize", choices=["none", "l2"], default="none")
    args = p.parse_args()

    model = str(args.model)
    dataset = str(args.dataset)
    seed = int(args.seed)
    normalize = str(args.normalize)

    k = load_k(dataset)
    z = load_best_z(model, dataset, seed)

    aug_dir = RESULTS_GRAY / model / dataset / f"seed{seed}" / "aug"
    X = pca_k(z, k=k, cache_dir=aug_dir / "pca_cache")

    if normalize == "l2":
        X = l2_normalize(X)

    comms = load_kmeans_communities(model, dataset, seed)
    ids = sorted(comms.keys())
    if not ids:
        print(f"[WARN] no communities >= {MIN_COMM_SIZE} for model={model}, dataset={dataset}, seed={seed}")
        return

    out_root = aug_dir / "gray_node"
    out_root.mkdir(parents=True, exist_ok=True)

    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            c1, c2 = ids[a], ids[b]
            C1, C2 = comms[c1], comms[c2]

            scores = compute_gray_scores(X, C1, C2)

            df = pd.DataFrame(scores, columns=["node_id", "score"])
            df["community_1"] = c1
            df["community_2"] = c2
            df["normalize"] = normalize 
            df["model"] = model
            df["dataset"] = dataset
            df["seed"] = seed

            out_path = out_root / f"pair_{c1}_{c2}.csv"
            df.to_csv(out_path, index=False)

    print(
        f"✅ Gray node scores saved: model={model}, dataset={dataset}, seed={seed}, "
        f"k={k}, normalize={normalize}, pairs={len(ids) * (len(ids) - 1) // 2}"
    )


if __name__ == "__main__":
    main()
