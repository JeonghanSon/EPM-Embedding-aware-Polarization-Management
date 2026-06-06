#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from filelock import FileLock

from src.utils.paths import RESULTS_BASE, DATA_SPLITS, DATA_META
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


def load_best_z(dataset: str, seed: int, model: str) -> torch.Tensor:
    p = RESULTS_BASE / model / dataset / f"seed{seed}" / "best_z.pt"
    z = torch.load(p, map_location="cpu")

    if not isinstance(z, torch.Tensor) or z.ndim != 2:
        raise TypeError(f"{dataset}: best_z.pt must be a 2D Tensor")

    return z.detach().cpu().float()


def read_train_edges(dataset: str) -> list[tuple[int, int, float]]:
    p = DATA_SPLITS / dataset / "train_edge_list.csv"
    df = pd.read_csv(p)[["source", "target", "weight"]].dropna()
    return [(int(u), int(v), float(w)) for u, v, w in df.itertuples(index=False)]


@torch.no_grad()
def pca_k(z: torch.Tensor, k: int) -> np.ndarray:
    _, H = z.shape
    k = int(min(k, H))

    if k <= 0:
        raise ValueError("k must be positive")

    mean = z.mean(dim=0)
    Zc = z - mean

    _, _, Vt = torch.linalg.svd(Zc, full_matrices=False)
    W = Vt[:k, :].T.contiguous()

    return ((z - mean) @ W).numpy()


def l2_norm(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (n + EPS)


def upsert_csv(path: Path, row: dict) -> None:
    cols = ["model", "dataset", "seed", "neg_scale", "k", "base_delta"]

    if path.exists():
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame(columns=cols)

    if not df.empty:
        mask = ~(
            (df["model"] == row["model"])
            & (df["dataset"] == row["dataset"])
            & (df["seed"] == row["seed"])
            & (df["neg_scale"] == row["neg_scale"])
        )
        df = df.loc[mask]

    df = pd.concat([df, pd.DataFrame([row], columns=cols)], ignore_index=True)
    df.to_csv(path, index=False)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", type=str, default="sgcn")
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
        z_path = RESULTS_BASE / args.model / dataset / f"seed{seed}" / "best_z.pt"

        if not z_path.exists():
            row = {
                "model": args.model,
                "dataset": dataset,
                "seed": seed,
                "neg_scale": neg_scale,
                "k": np.nan,
                "base_delta": np.nan,
            }
            with csv_lock:
                upsert_csv(out_csv, row)
            print(f"[WARN] missing z: {dataset} seed={seed}")
            continue

        try:
            k = load_k(dataset)
            z = load_best_z(dataset, seed, args.model)

            X = pca_k(z, k=k)
            X = l2_norm(X)

            edge_list = read_train_edges(dataset)
            delta = delta_solver_multi(edge_list, X, pos=1.0, neg=neg_scale)

            row = {
                "model": args.model,
                "dataset": dataset,
                "seed": seed,
                "neg_scale": neg_scale,
                "k": int(min(k, z.shape[1])),
                "base_delta": float(delta),
            }

            with csv_lock:
                upsert_csv(out_csv, row)

            print(f"base Δ: {dataset} seed={seed} neg={neg_scale} k={row['k']} -> {delta}")

        except Exception as e:
            row = {
                "model": args.model,
                "dataset": dataset,
                "seed": seed,
                "neg_scale": neg_scale,
                "k": np.nan,
                "base_delta": np.nan,
            }
            with csv_lock:
                upsert_csv(out_csv, row)

            print(f"[WARN] failed: {dataset} seed={seed} ({e})")

    print(f"saved: {out_csv}")


if __name__ == "__main__":
    main()
