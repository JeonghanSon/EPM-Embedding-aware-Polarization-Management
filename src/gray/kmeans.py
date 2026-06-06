# src/gray/kmeans.py
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans

from src.utils.paths import RESULTS_BASE, RESULTS_GRAY, DATA_META


def _load_k(dataset: str) -> int:
    p = DATA_META / dataset / "num_communities.json"
    with open(p, "r", encoding="utf-8") as f:
        meta = json.load(f)
    k = meta.get("mode", meta.get("avg", None))
    if k is None:
        raise RuntimeError(f"{dataset}: num_communities.json missing mode/avg")
    return int(k)


def _load_best_z(model: str, dataset: str, seed: int) -> np.ndarray:
    p = RESULTS_BASE / model / dataset / f"seed{seed}" / "best_z.pt"
    z = torch.load(p, map_location="cpu")
    if isinstance(z, dict):
        z = next(iter(z.values()))
    if not isinstance(z, torch.Tensor) or z.ndim != 2:
        raise TypeError(
            f"{dataset}: best_z.pt must be 2D Tensor "
            f"(got {type(z)}, shape={getattr(z, 'shape', None)})"
        )
    return z.detach().cpu().float().numpy()


def _save_nodewise(labels: np.ndarray, out_path: Path):
    df = pd.DataFrame({
        "node_id": np.arange(len(labels), dtype=int),
        "community": labels.astype(int),
    })
    df.to_csv(out_path, index=False)


def _save_groupwise(labels: np.ndarray, out_path: Path):
    comm = defaultdict(list)
    for i, c in enumerate(labels.tolist()):
        comm[int(c)].append(i)
    records = [{"community_id": cid, "nodes": str(nodes)} for cid, nodes in comm.items()]
    pd.DataFrame(records).to_csv(out_path, index=False)


def run(model: str, dataset: str, seed: int, random_state: int = 42):
    k = _load_k(dataset)
    X = _load_best_z(model, dataset, seed)

    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = km.fit_predict(X)

    out_dir = RESULTS_GRAY / model / dataset / f"seed{seed}" / "aug"
    out_dir.mkdir(parents=True, exist_ok=True)

    _save_nodewise(labels, out_dir / "kmeans_node.csv")
    _save_groupwise(labels, out_dir / "kmeans_community.csv")

    print(f"✅ kmeans saved: model={model} dataset={dataset} seed={seed} k={k} -> {out_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="sgcn")
    p.add_argument("--dataset", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--random_state", type=int, default=42)
    args = p.parse_args()

    run(args.model, args.dataset, args.seed, args.random_state)


if __name__ == "__main__":
    main()
