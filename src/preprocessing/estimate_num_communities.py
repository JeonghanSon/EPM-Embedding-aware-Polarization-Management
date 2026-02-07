from __future__ import annotations

import json
from collections import Counter
import numpy as np
import pandas as pd

from src.utils.paths import DATA_SPLITS, DATA_META
from src.preprocessing.community_detection.run_signed_louvain import run_signed_louvain


MIN_COMM_SIZE = 30
MIN_COMM_SIZE_BY_DATASET = {
    "Slashdot": 500,   # adjust if needed
    "Epinions": 500,   # adjust if needed
}


def _parse_comm_sizes(csv_path) -> list[int]:
    comm_df = pd.read_csv(csv_path)
    if "nodes" not in comm_df.columns:
        raise RuntimeError(f"Invalid communities.csv (missing 'nodes'): {csv_path}")

    sizes = []
    for v in comm_df["nodes"]:
        if isinstance(v, str):
            parts = [x.strip() for x in v.split(",") if x.strip()]
            sizes.append(len(parts))
        else:
            sizes.append(0 if pd.isna(v) else 1)
    return sizes


def _latest_undirected_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    df = df[["source", "target", "weight"]].copy()
    df = df.dropna(subset=["source", "target", "weight"]).copy()

    df["source"] = pd.to_numeric(df["source"], errors="coerce")
    df["target"] = pd.to_numeric(df["target"], errors="coerce")
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
    df = df.dropna(subset=["source", "target", "weight"]).copy()

    df["source"] = df["source"].astype(int)
    df["target"] = df["target"].astype(int)
    df["weight"] = df["weight"].astype(int)

    df = df[(df["weight"] != 0) & (df["source"] != df["target"])].reset_index(drop=True)

    s = df["source"].to_numpy()
    t = df["target"].to_numpy()
    df["source"] = np.minimum(s, t)
    df["target"] = np.maximum(s, t)

    df["__idx__"] = np.arange(len(df), dtype=np.int64)
    df = df.drop_duplicates(subset=["source", "target"], keep="last")
    df = df.sort_values("__idx__", kind="mergesort").drop(columns="__idx__").reset_index(drop=True)
    return df


def estimate_num_communities(
    dataset: str,
    n_runs: int = 10,
    base_seed: int = 0,
    min_comm_size: int = MIN_COMM_SIZE,
):
    train_path = DATA_SPLITS / dataset / "train_edge_list.csv"
    min_comm_size = int(MIN_COMM_SIZE_BY_DATASET.get(dataset, min_comm_size))
    if not train_path.exists():
        raise FileNotFoundError(f"Train split not found: {train_path}")

    df_train = pd.read_csv(train_path)
    if len(df_train) == 0:
        print(f"[{dataset}] train_edge_list.csv is empty.")
        return None

    df_snap = _latest_undirected_snapshot(df_train)

    max_node = int(max(df_snap["source"].max(), df_snap["target"].max()))
    nb_nodes = max_node + 1

    edges = list(
        zip(
            df_snap["source"].astype(int),
            df_snap["target"].astype(int),
            np.sign(df_snap["weight"]).astype(float),
        )
    )

    save_dir = DATA_SPLITS / dataset / "louvain"
    save_dir.mkdir(parents=True, exist_ok=True)

    meta_dir = DATA_META / dataset
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta_path = meta_dir / "num_communities.json"

    num_list = []

    for i in range(n_runs):
        seed = base_seed + i

        run_signed_louvain(
            edge_list=edges,
            nb_nodes=nb_nodes,
            save_dir=str(save_dir),
            seed=seed,
        )

        csv_path = save_dir / "communities.csv"
        if not csv_path.exists():
            raise RuntimeError(f"[{dataset}] communities.csv not found: {csv_path}")

        kept = save_dir / f"communities_seed{seed}.csv"
        try:
            csv_path.replace(kept)  # move
        except Exception:
            kept = csv_path  # fallback (may be overwritten by subsequent runs)

        sizes = _parse_comm_sizes(kept)
        num_valid = sum(1 for s in sizes if s >= min_comm_size)
        num_list.append(num_valid)

        print(
            f"{dataset} run {i+1}/{n_runs} seed={seed} -> "
            f"total={len(sizes)} valid(>={min_comm_size})={num_valid}"
        )

    avg_num = float(sum(num_list) / len(num_list))
    mode_num = int(Counter(num_list).most_common(1)[0][0])

    meta = {
        "dataset": dataset,
        "train_path": str(train_path),
        "policy": "undirected; drop duplicates and keep last edge",
        "min_comm_size": int(min_comm_size),
        "n_runs": int(n_runs),
        "base_seed": int(base_seed),
        "nb_nodes": int(nb_nodes),
        "edges_train_raw": int(len(df_train)),
        "edges_snapshot": int(len(df_snap)),
        "runs": num_list,
        "avg": avg_num,
        "mode": mode_num,
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[META] Saved -> {meta_path}")
    return meta


def main():
    datasets = ["bitcoinalpha", "bitcoinotc", "wiki-RfA", "wiki-Elec" "Slashdot", "Epinions"]
    for d in datasets:
        estimate_num_communities(d)


if __name__ == "__main__":
    main()
