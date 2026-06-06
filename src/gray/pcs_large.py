#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

from src.utils.paths import DATA_META, DATA_SPLITS, RESULTS_BASE, RESULTS_GRAY

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
        raise TypeError(f"{p}: best_z.pt must be 2D Tensor")

    return z.detach().cpu().float()


def read_train_edges(dataset: str) -> list[tuple[int, int, float]]:
    p = DATA_SPLITS / dataset / "train_edge_list.csv"
    df = pd.read_csv(p)[["source", "target", "weight"]].dropna()
    return [(int(u), int(v), float(w)) for u, v, w in df.itertuples(index=False)]


def dedup_keep_last_undirected(
    edge_list: list[tuple[int, int, float]]
) -> list[tuple[int, int, float]]:
    last: dict[tuple[int, int], float] = {}

    for u, v, w in edge_list:
        u = int(u)
        v = int(v)
        w = float(w)

        if u == v or w == 0:
            continue

        a, b = (u, v) if u < v else (v, u)
        last[(a, b)] = w

    return [(a, b, w) for (a, b), w in last.items()]


def load_kmeans_communities(
    model: str,
    dataset: str,
    seed: int,
    min_comm_size: int,
) -> dict[int, list[int]]:
    p = RESULTS_GRAY / model / dataset / f"seed{seed}" / "aug" / "kmeans_community.csv"
    df = pd.read_csv(p)

    out: dict[int, list[int]] = {}

    for _, r in df.iterrows():
        cid = int(r["community_id"])

        try:
            nodes = ast.literal_eval(str(r["nodes"]))
        except Exception:
            nodes = []

        if not isinstance(nodes, list):
            nodes = [nodes]

        nodes = [int(x) for x in nodes]

        if len(nodes) >= int(min_comm_size):
            out[cid] = nodes

    return out


@torch.no_grad()
@torch.no_grad()
def pca_k_l2(z: torch.Tensor, k: int, cache_dir: Path) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)

    _, h = z.shape
    k = int(min(k, h))

    if k <= 0:
        raise ValueError("k must be positive")

    cache = cache_dir / f"pca_k{k}_H{h}.pt"

    if cache.exists():
        ck = torch.load(cache, map_location="cpu")
        mean = ck["mean"]
        W = ck["W"]
    else:
        mean = z.mean(dim=0)
        Zc = z - mean
        _, _, Vt = torch.linalg.svd(Zc, full_matrices=False)
        W = Vt[:k, :].T.contiguous()
        torch.save({"mean": mean, "W": W}, cache)

    X = ((z - mean) @ W).numpy()
    n = np.linalg.norm(X, axis=1, keepdims=True)

    return X / (n + EPS)


def centroid_distance_pruning(
    X: np.ndarray,
    comms: dict[int, list[int]],
    top_pairs: int,
) -> list[tuple[int, int, float]]:
    ids = sorted(comms.keys())
    centroids = {cid: X[comms[cid]].mean(axis=0) for cid in ids}

    scores = []

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            c1, c2 = ids[i], ids[j]
            score = float(np.linalg.norm(centroids[c1] - centroids[c2]))
            scores.append((c1, c2, score))

    scores.sort(key=lambda t: t[2], reverse=True)
    return scores[: max(int(top_pairs), 0)]


def build_laplacian(
    edge_list: list[tuple[int, int, float]],
    num_nodes: int,
    neg_scale: float,
    pos_scale: float = 1.0,
) -> sp.csr_matrix:
    rows, cols, data = [], [], []

    for u, v, w in edge_list:
        u = int(u)
        v = int(v)

        if u == v:
            continue

        if not (0 <= u < num_nodes and 0 <= v < num_nodes):
            continue

        w_adj = float(pos_scale if w > 0 else neg_scale)

        rows.extend([u, v])
        cols.extend([v, u])
        data.extend([w_adj, w_adj])

    A = sp.coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes)).tocsr()
    deg = np.asarray(A.sum(axis=1)).ravel()

    return (sp.diags(deg) - A).tocsr()


def delta_solver_axes_mean_union(
    L: sp.csr_matrix,
    X: np.ndarray,
    nodes: list[int],
    rtol: float = 1e-5,
    atol: float = 1e-8,
    maxiter: int = 5000,
) -> float:
    X = np.asarray(X, dtype=np.float64)

    n, k = X.shape
    if k <= 0 or len(nodes) == 0:
        return 0.0

    nodes_arr = np.asarray(nodes, dtype=np.int64)

    Xu = X[nodes_arr]
    Vu = Xu - Xu.mean(axis=0, keepdims=True)

    ones = np.ones(n, dtype=np.float64)

    def matvec(x: np.ndarray) -> np.ndarray:
        return (L @ x) + (x.mean() * ones)

    Aop = spla.LinearOperator((n, n), matvec=matvec, dtype=np.float64)

    inv_diag = 1.0 / np.maximum(L.diagonal().astype(np.float64), 1e-12)
    Mop = spla.LinearOperator((n, n), matvec=lambda x: inv_diag * x, dtype=np.float64)

    s = 0.0

    for t in range(k):
        b = np.zeros(n, dtype=np.float64)
        b[nodes_arr] = Vu[:, t]

        y, info = spla.cg(Aop, b, M=Mop, rtol=rtol, atol=atol, maxiter=maxiter)

        if info != 0:
            return float("nan")

        y = y - y.mean()
        q = float(b.T @ y)

        if q < 0 and q > -1e-9:
            q = 0.0

        s += q

    m = s / float(k)

    if m < 0 and m > -1e-9:
        m = 0.0

    return float(np.sqrt(max(m, 0.0)))


def empty_output(out_csv: Path, model: str, dataset: str, seed: int, neg_scale: float) -> None:
    pd.DataFrame(
        columns=[
            "model",
            "dataset",
            "seed",
            "neg_scale",
            "community_1",
            "community_2",
            "size_c1",
            "size_c2",
            "prune_score",
            "delta",
        ]
    ).to_csv(out_csv, index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="sgcn")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--neg_scale", type=float, default=0.1)
    ap.add_argument("--min_comm_size", type=int, default=1000)
    ap.add_argument("--top_pairs", type=int, default=50)
    ap.add_argument("--rtol", type=float, default=1e-5)
    ap.add_argument("--atol", type=float, default=1e-8)
    ap.add_argument("--maxiter", type=int, default=5000)
    args = ap.parse_args()

    model = str(args.model)
    dataset = str(args.dataset)
    seed = int(args.seed)
    neg_scale = float(args.neg_scale)
    min_comm_size = int(args.min_comm_size)
    top_pairs = int(args.top_pairs)

    k = load_k(dataset)
    z = load_best_z(model, dataset, seed)

    aug_dir = RESULTS_GRAY / model / dataset / f"seed{seed}" / "aug"
    out_dir = aug_dir / "pcs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "delta_pairs.csv"

    X = pca_k_l2(
        z,
        k,
        cache_dir=aug_dir / "pca_cache",
    )

    comms = load_kmeans_communities(
        model=model,
        dataset=dataset,
        seed=seed,
        min_comm_size=min_comm_size,
    )

    if len(comms) < 2:
        empty_output(out_csv, model, dataset, seed, neg_scale)
        print(
            f"[WARN] not enough communities after size filter "
            f"(>= {min_comm_size}): model={model} dataset={dataset} seed={seed}"
        )
        return

    candidates = centroid_distance_pruning(X, comms, top_pairs=top_pairs)

    if len(candidates) == 0:
        empty_output(out_csv, model, dataset, seed, neg_scale)
        print(f"[WARN] no candidate pairs: model={model} dataset={dataset} seed={seed}")
        return

    edges = read_train_edges(dataset)
    dedup = dedup_keep_last_undirected(edges)

    L = build_laplacian(
        dedup,
        num_nodes=X.shape[0],
        neg_scale=neg_scale,
        pos_scale=1.0,
    )

    rows = []

    for c1, c2, prune_score in candidates:
        nodes = comms[int(c1)] + comms[int(c2)]

        delta = delta_solver_axes_mean_union(
            L=L,
            X=X,
            nodes=nodes,
            rtol=float(args.rtol),
            atol=float(args.atol),
            maxiter=int(args.maxiter),
        )

        rows.append(
            {
                "model": model,
                "dataset": dataset,
                "seed": seed,
                "neg_scale": neg_scale,
                "community_1": int(c1),
                "community_2": int(c2),
                "size_c1": len(comms[int(c1)]),
                "size_c2": len(comms[int(c2)]),
                "prune_score": float(prune_score),
                "delta": float(delta),
            }
        )

    pd.DataFrame(rows).sort_values("delta", ascending=False).to_csv(out_csv, index=False)

    print(
        f"✅ PCS_LARGE(pruned) saved: {out_csv} | "
        f"model={model} dataset={dataset} seed={seed} "
        f"min_size>={min_comm_size} -> {len(comms)} comms, "
        f"top_pairs={top_pairs} -> {len(rows)} pairs"
    )


if __name__ == "__main__":
    main()
