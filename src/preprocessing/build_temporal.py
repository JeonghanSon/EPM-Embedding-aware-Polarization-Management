from __future__ import annotations

"""
Build a clean edge list under `data/build/<dataset>/edge_list.csv` from
`data/interim/<dataset>/edge_list.csv`.

- Temporal datasets: keep `timestamp` and assign `event_id`.
- Static datasets: deduplicate undirected pairs and keep the last occurrence.
"""

import numpy as np
import pandas as pd

from src.utils.paths import DATA_INTERIM, DATA_BUILD


def _ensure_cols(df: pd.DataFrame, cols: list[str], dataset: str):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"{dataset}: missing columns {miss}")


def _remap_nodes_first_seen(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    node2id = {}
    next_id = 0

    src = df["source"].to_numpy()
    tgt = df["target"].to_numpy()
    new_src = np.empty(len(df), dtype=np.int64)
    new_tgt = np.empty(len(df), dtype=np.int64)

    for i in range(len(df)):
        a = src[i]
        b = tgt[i]
        if a not in node2id:
            node2id[a] = next_id
            next_id += 1
        if b not in node2id:
            node2id[b] = next_id
            next_id += 1
        new_src[i] = node2id[a]
        new_tgt[i] = node2id[b]

    out = df.copy()
    out["source"] = new_src
    out["target"] = new_tgt
    return out, next_id


def build(dataset_name: str) -> None:
    inp = DATA_INTERIM / dataset_name / "edge_list.csv"
    out_dir = DATA_BUILD / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    outp = out_dir / "edge_list.csv"

    df = pd.read_csv(inp)
    _ensure_cols(df, ["source", "target", "weight"], dataset_name)

    is_temporal = "timestamp" in df.columns

    if is_temporal:
        _ensure_cols(df, ["timestamp"], dataset_name)
        df = df.dropna(subset=["source", "target", "weight", "timestamp"]).copy()

        df["source"] = df["source"].astype(str)
        df["target"] = df["target"].astype(str)
        df["weight"] = pd.to_numeric(df["weight"], errors="coerce").astype("Int64")
        df = df.dropna(subset=["weight"]).copy()
        df["weight"] = df["weight"].astype(int)

        df = df[(df["weight"] != 0) & (df["source"] != df["target"])].copy()
        df = df.sort_values("timestamp").reset_index(drop=True)
        df["event_id"] = np.arange(len(df), dtype=np.int64)

        df, n_nodes = _remap_nodes_first_seen(df)
        df.to_csv(outp, index=False)
        print(f"[BUILD] {dataset_name} temporal -> {outp} (nodes={n_nodes}, edges={len(df)})")
        return

    # static
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
    df = df.sort_values("__idx__", kind="mergesort").reset_index(drop=True)
    df = df.drop(columns=["__idx__"])

    df["source"] = df["source"].astype(str)
    df["target"] = df["target"].astype(str)
    df, n_nodes = _remap_nodes_first_seen(df)

    df = df[["source", "target", "weight"]]
    df.to_csv(outp, index=False)
    print(f"[BUILD] {dataset_name} static -> {outp} (nodes={n_nodes}, edges={len(df)})")


def run_all() -> None:
    temporal = ["bitcoinalpha", "bitcoinotc", "wiki-RfA", "wiki-Elec"]
    static = ["Slashdot", "Epinions"]
    for name in temporal + static:
        build(name)


if __name__ == "__main__":
    run_all()
