#!/usr/bin/env python3
# src/train/train_gray.py
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from filelock import FileLock

from src.utils.seed import set_seed
from src.utils.paths import RESULTS_GRAY
from src.train.train_sgcn import train_sgcn_for_signlink_class

MODEL_REGISTRY = {
    "sgcn": train_sgcn_for_signlink_class,
}


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


def upsert_best_csv(path: Path, row: dict, key_cols: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
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
    p = argparse.ArgumentParser()

    p.add_argument("--model", type=str, default="sgcn", choices=["sgcn"])
    p.add_argument("--dataset", required=True)
    p.add_argument("--seed", type=int, required=True)

    # connect config (must match gray_connect.py)
    p.add_argument("--gray_mode", choices=["min", "avg", "max"], required=True)
    p.add_argument("--scope", choices=["global", "local"], required=True)

    p.add_argument("--pair_selector", choices=["threshold", "topk"], required=True)
    p.add_argument("--minmax", type=float, required=True)
    p.add_argument("--max_degree", type=float, required=True)
    p.add_argument("--topk", type=int, default=None)

    p.add_argument("--setting", choices=["none", "l2"], required=True)
    p.add_argument("--edge_scale", type=float, required=True)

    # Default to a single configuration (reviewer-friendly),
    # while still allowing multiple values via CLI.
    p.add_argument("--embedding_dims", nargs="+", type=int, default=[64])
    p.add_argument("--num_layers", nargs="+", type=int, default=[2])
    p.add_argument("--lrs", nargs="+", type=float, default=[0.01])

    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--patience", type=int, default=10)

    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    set_seed(int(args.seed))

    model = str(args.model)
    dataset = str(args.dataset)
    seed = int(args.seed)

    if model not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model: {model}")

    out_dir = gray_save_dir(
        model=model,
        dataset=dataset,
        seed=seed,
        gray_mode=args.gray_mode,
        scope=args.scope,
        pair_selector=args.pair_selector,
        minmax_th=float(args.minmax),
        max_degree=float(args.max_degree),
        topk=args.topk,
        setting=args.setting,
        edge_scale=float(args.edge_scale),
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    aug_train_csv = out_dir / "train_edge_list_aug.csv"
    if not aug_train_csv.exists():
        raise FileNotFoundError(f"Missing augmented train edge list: {aug_train_csv}")

    # outputs (base-style)
    best_z = out_dir / "best_z.pt"
    best_model = out_dir / "best_model.pt"
    best_metrics = out_dir / "best_metrics.json"
    best_config = out_dir / "best_config.json"

    if best_z.exists() and best_model.exists() and best_metrics.exists() and best_config.exists() and not args.overwrite:
        print(f"⏭️ Already exists (skip): {out_dir}  (use --overwrite)")
        return

    train_tag = (
        f"gray_{model}_{args.gray_mode}_{args.scope}_{args.pair_selector}"
        f"_mm{int(round(args.minmax * 100))}"
        f"_deg{('INF' if np.isinf(args.max_degree) or args.max_degree >= 1e9 else int(args.max_degree))}"
        f"_topk{int(args.topk) if args.topk is not None else 0}"
        f"_{args.setting}"
        f"_esc{int(round(args.edge_scale * 100))}"
    )

    tmp_root = out_dir / "_tmp_grid"
    if tmp_root.exists():
        shutil.rmtree(tmp_root, ignore_errors=True)
    tmp_root.mkdir(parents=True, exist_ok=True)

    best_row = None
    best_val_f1 = -1.0
    train_fn = MODEL_REGISTRY[model]

    try:
        for dim in list(args.embedding_dims):
            for L in list(args.num_layers):
                for lr in list(args.lrs):
                    run_name = f"dim{int(dim)}_L{int(L)}_lr{lr}"
                    run_dir = tmp_root / run_name
                    if run_dir.exists():
                        shutil.rmtree(run_dir, ignore_errors=True)
                    run_dir.mkdir(parents=True, exist_ok=True)

                    print(f"[GRAY-TRAIN] model={model} {dataset} seed={seed} {run_name}")

                    metrics, artifacts = train_fn(
                        dataset=dataset,
                        save_dir=run_dir,
                        embedding_dim=int(dim),
                        num_layers=int(L),
                        epochs=int(args.epochs),
                        lr=float(lr),
                        patience=int(args.patience),
                        seed=seed,
                        train_edge_csv=aug_train_csv,
                        train_tag=train_tag,
                        model_name=model,
                    )

                    row = {
                        "model": model,
                        "dataset": dataset,
                        "seed": seed,
                        "gray_mode": args.gray_mode,
                        "scope": args.scope,
                        "pair_selector": args.pair_selector,
                        "minmax": float(args.minmax),
                        "max_degree": float(args.max_degree),
                        "topk": int(args.topk) if args.topk is not None else None,
                        "setting": args.setting,
                        "edge_scale": float(args.edge_scale),
                        "train_tag": train_tag,
                        "embedding_dim": int(dim),
                        "num_layers": int(L),
                        "lr": float(lr),
                        "val_accuracy": metrics.get("val_accuracy", np.nan),
                        "val_f1": metrics.get("val_f1", np.nan),
                        "test_accuracy": metrics.get("test_accuracy", np.nan),
                        "test_f1": metrics.get("test_f1", np.nan),
                    }

                    val_f1 = row["val_f1"]
                    if val_f1 is not None and not pd.isna(val_f1) and float(val_f1) > best_val_f1:
                        best_val_f1 = float(val_f1)
                        best_row = row

                        z_src = Path(artifacts["z_path"])
                        m_src = Path(artifacts["model_path"])
                        j_src = Path(artifacts["metrics_path"])

                        shutil.copyfile(z_src, best_z)
                        shutil.copyfile(m_src, best_model)
                        shutil.copyfile(j_src, best_metrics)

                        cfg = {
                            "model": model,
                            "dataset": dataset,
                            "seed": seed,
                            "gray_mode": args.gray_mode,
                            "scope": args.scope,
                            "pair_selector": args.pair_selector,
                            "minmax": float(args.minmax),
                            "max_degree": float(args.max_degree),
                            "topk": int(args.topk) if args.topk is not None else None,
                            "setting": args.setting,
                            "edge_scale": float(args.edge_scale),
                            "embedding_dim": int(dim),
                            "num_layers": int(L),
                            "lr": float(lr),
                            "epochs": int(args.epochs),
                            "patience": int(args.patience),
                            "train_tag": train_tag,
                            "best_val_f1": float(best_val_f1),
                        }
                        with open(best_config, "w", encoding="utf-8") as f:
                            json.dump(cfg, f, indent=2)

                    shutil.rmtree(run_dir, ignore_errors=True)

        if best_row is None:
            raise RuntimeError(f"No valid best found (all val_f1 NaN?) for model={model} dataset={dataset} seed={seed} at {out_dir}")

    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    best_csv = RESULTS_GRAY / "best_embeddings.csv"
    key_cols = [
        "model", "dataset", "seed",
        "gray_mode", "scope", "pair_selector",
        "minmax", "max_degree", "topk", "setting", "edge_scale",
    ]
    upsert_best_csv(best_csv, best_row, key_cols)

    print(f"✅ Gray train done: {out_dir}")
    print(f"🏁 Best val_f1={best_val_f1:.6f}")
    print(f"   - best_z: {best_z}")
    print(f"   - best_model: {best_model}")
    print(f"   - best_metrics: {best_metrics}")
    print(f"   - best_config: {best_config}")


if __name__ == "__main__":
    main()
