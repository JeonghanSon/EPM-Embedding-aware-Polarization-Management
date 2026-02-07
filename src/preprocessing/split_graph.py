# src/preprocessing/split_graph.py
from __future__ import annotations

import json
from collections import deque
import numpy as np
import pandas as pd

from src.utils.paths import DATA_BUILD, DATA_SPLITS


SEED = 42
RATIOS = (0.7, 0.15, 0.15)


def _clean_common(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["source", "target", "weight"]).copy()
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
    df = df.dropna(subset=["weight"]).copy()

    # Keep only the sign (convert to an unweighted signed graph).
    df["weight"] = np.sign(df["weight"]).astype(int)
    df = df[df["weight"] != 0].copy()

    df["source"] = pd.to_numeric(df["source"], errors="coerce")
    df["target"] = pd.to_numeric(df["target"], errors="coerce")
    df = df.dropna(subset=["source", "target"]).copy()
    df["source"] = df["source"].astype(int)
    df["target"] = df["target"].astype(int)

    df = df[df["source"] != df["target"]].reset_index(drop=True)
    return df


def _gcc_nodes_from_edges(src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    if len(src) == 0:
        return np.array([], dtype=np.int64)

    n = int(max(src.max(), tgt.max())) + 1
    parent = np.arange(n, dtype=np.int64)
    size = np.ones(n, dtype=np.int64)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in zip(src, tgt):
        ra, rb = find(int(a)), find(int(b))
        if ra == rb:
            continue
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]

    roots = np.array([find(i) for i in range(n)], dtype=np.int64)
    gcc_root = np.bincount(roots).argmax()
    return np.where(roots == gcc_root)[0]


def _remap_by_nodes(df: pd.DataFrame, nodes: np.ndarray) -> tuple[pd.DataFrame, dict, int]:
    nodes = np.array(nodes, dtype=np.int64)
    nodes_sorted = np.sort(nodes)
    node_map = {int(u): i for i, u in enumerate(nodes_sorted)}
    out = df.copy()
    out["source"] = out["source"].map(node_map)
    out["target"] = out["target"].map(node_map)
    out = out.dropna(subset=["source", "target"]).copy()
    out["source"] = out["source"].astype(int)
    out["target"] = out["target"].astype(int)
    return out.reset_index(drop=True), node_map, len(nodes_sorted)


def _write_split(dataset: str, train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame, meta: dict):
    out_dir = DATA_SPLITS / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    train.to_csv(out_dir / "train_edge_list.csv", index=False)
    valid.to_csv(out_dir / "valid_edge_list.csv", index=False)
    test.to_csv(out_dir / "test_edge_list.csv", index=False)

    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(
        f"✅ {dataset} saved: "
        f"train={len(train)}, valid={len(valid)}, test={len(test)}, nodes(train)={meta['n_train_nodes']}"
    )


def _dedup_keep_last_by_order(df: pd.DataFrame, order_col: str = "__order__") -> pd.DataFrame:
    """
    Deduplication applied *after* splitting.

    - Convert (u, v) into an undirected canonical form.
    - Keep the last occurrence within each split according to `order_col`
      (which encodes the original time-sorted order).
    - Do not coerce `timestamp` into numeric values to avoid format issues
      (e.g., Wiki-Elec).
    """
    if len(df) == 0:
        return df.reset_index(drop=True)

    out = df.copy()

    # undirected canonical
    s, t = out["source"].to_numpy(), out["target"].to_numpy()
    out["source"] = np.minimum(s, t)
    out["target"] = np.maximum(s, t)

    # keep last within split (time order = order_col)
    out = out.sort_values(order_col, kind="mergesort").reset_index(drop=True)
    out = out.drop_duplicates(subset=["source", "target"], keep="last")
    out = out.sort_values(order_col, kind="mergesort").reset_index(drop=True)

    return out


def split_static(dataset: str, seed: int = SEED, ratios: tuple[float, float, float] = RATIOS):
    inp = DATA_BUILD / dataset / "edge_list.csv"
    df = pd.read_csv(inp)
    df = _clean_common(df)

    # undirected canonical form
    s, t = df["source"].to_numpy(), df["target"].to_numpy()
    df["source"] = np.minimum(s, t)
    df["target"] = np.maximum(s, t)

    # keep-last duplicates
    df["__idx__"] = np.arange(len(df), dtype=np.int64)
    df = df.drop_duplicates(subset=["source", "target"], keep="last")
    df = df.sort_values("__idx__", kind="mergesort").drop(columns="__idx__").reset_index(drop=True)

    # GCC on whole graph
    gcc_nodes = _gcc_nodes_from_edges(df["source"].to_numpy(), df["target"].to_numpy())
    df = df[df["source"].isin(gcc_nodes) & df["target"].isin(gcc_nodes)].reset_index(drop=True)

    # remap GCC nodes -> 0..N-1 (so spanning tree can cover all)
    df, _, n_nodes = _remap_by_nodes(df, np.unique(df[["source", "target"]].to_numpy()))

    # build adjacency + edge id
    src = df["source"].to_numpy()
    tgt = df["target"].to_numpy()
    n = n_nodes
    adj = [[] for _ in range(n)]
    edge_id = {}
    for i, (a, b) in enumerate(zip(src, tgt)):
        a, b = int(a), int(b)
        adj[a].append(b)
        adj[b].append(a)
        edge_id[(min(a, b), max(a, b))] = i

    # spanning tree edges (BFS)
    visited = np.zeros(n, dtype=bool)
    visited[0] = True
    q = deque([0])
    tree_edges = []

    while q:
        u = q.popleft()
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                q.append(v)
                tree_edges.append(edge_id[(min(u, v), max(u, v))])

    tree_edges = np.array(tree_edges, dtype=np.int64)
    m = len(df)
    n_train = int(round(m * ratios[0]))
    n_valid = int(round(m * ratios[1]))
    n_test = m - n_train - n_valid

    if n_train < len(tree_edges):
        raise RuntimeError(f"{dataset}: train too small for spanning tree (train={n_train}, tree={len(tree_edges)})")

    rng = np.random.default_rng(seed)
    all_idx = np.arange(m, dtype=np.int64)
    tree_set = set(tree_edges.tolist())
    rest = np.array([i for i in all_idx if i not in tree_set], dtype=np.int64)
    rng.shuffle(rest)

    need = n_train - len(tree_edges)
    train_idx = np.concatenate([tree_edges, rest[:need]])
    remain = rest[need:]
    valid_idx = remain[:n_valid]
    test_idx = remain[n_valid : n_valid + n_test]

    train = df.iloc[train_idx].reset_index(drop=True)
    valid = df.iloc[valid_idx].reset_index(drop=True)
    test = df.iloc[test_idx].reset_index(drop=True)

    # transductive + final remap by train nodes
    train_nodes = np.unique(train[["source", "target"]].to_numpy())
    valid = valid[valid["source"].isin(train_nodes) & valid["target"].isin(train_nodes)].reset_index(drop=True)
    test = test[test["source"].isin(train_nodes) & test["target"].isin(train_nodes)].reset_index(drop=True)

    train, _, n_train_nodes = _remap_by_nodes(train, train_nodes)
    valid, _, _ = _remap_by_nodes(valid, train_nodes)
    test, _, _ = _remap_by_nodes(test, train_nodes)

    meta = {
        "dataset": dataset,
        "type": "static",
        "seed": seed,
        "ratios": list(ratios),
        "edges_total_after_gcc": int(m),
        "edges_train": int(len(train)),
        "edges_valid": int(len(valid)),
        "edges_test": int(len(test)),
        "n_train_nodes": int(n_train_nodes),
        "dropped_valid_test_due_to_transductive": True,
    }
    _write_split(dataset, train, valid, test, meta)


def split_temporal_split_then_dedup(dataset: str, seed: int = SEED, ratios: tuple[float, float, float] = RATIOS):
    """
    Policy:
    1) Time-based split by `timestamp` (70/15/15). Do NOT force numeric parsing of timestamps.
    2) After splitting, deduplicate edges *within each split* and keep the latest one
       (i.e., the last in the time-sorted order).
    3) Transductive setting: keep only nodes observed in the training split for val/test.
    4) Ensure connectivity by keeping only the GCC of the training split.
    """
    inp = DATA_BUILD / dataset / "edge_list.csv"
    df = pd.read_csv(inp)
    df = _clean_common(df)

    if "timestamp" not in df.columns:
        raise ValueError(f"{dataset}: requires 'timestamp' column")

    # Sort by timestamp without forcing numeric conversion (robust to mixed formats).
    df = df.dropna(subset=["timestamp"]).copy()
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    # Assign an order index to preserve the time-sorted order for per-split deduplication.

    df["__order__"] = np.arange(len(df), dtype=np.int64)

    # Keep only the GCC of the full graph to remove small noisy components.
    src_all = df["source"].to_numpy()
    tgt_all = df["target"].to_numpy()
    gcc_nodes_all = _gcc_nodes_from_edges(src_all, tgt_all)
    df = df[df["source"].isin(gcc_nodes_all) & df["target"].isin(gcc_nodes_all)].reset_index(drop=True)

    m0 = len(df)
    n_train0 = int(round(m0 * ratios[0]))
    n_valid0 = int(round(m0 * ratios[1]))
    n_test0 = m0 - n_train0 - n_valid0

    train_raw = df.iloc[:n_train0].reset_index(drop=True)
    valid_raw = df.iloc[n_train0 : n_train0 + n_valid0].reset_index(drop=True)
    test_raw = df.iloc[n_train0 + n_valid0 :].reset_index(drop=True)

    # Deduplicate within each split and keep the last edge by `__order__`.
    train1 = _dedup_keep_last_by_order(train_raw, order_col="__order__")
    valid1 = _dedup_keep_last_by_order(valid_raw, order_col="__order__")
    test1 = _dedup_keep_last_by_order(test_raw, order_col="__order__")

    # Ensure training connectivity by keeping only the GCC of the training split.
    if len(train1) > 0:
        gcc_nodes_tr = _gcc_nodes_from_edges(train1["source"].to_numpy(), train1["target"].to_numpy())
        train2 = train1[train1["source"].isin(gcc_nodes_tr) & train1["target"].isin(gcc_nodes_tr)].reset_index(drop=True)
    else:
        train2 = train1

    # Transductive: keep only nodes seen in training for valid/test.
    train_nodes = (
        np.unique(train2[["source", "target"]].to_numpy()) if len(train2) > 0 else np.array([], dtype=np.int64)
    )
    valid2 = valid1[valid1["source"].isin(train_nodes) & valid1["target"].isin(train_nodes)].reset_index(drop=True)
    test2 = test1[test1["source"].isin(train_nodes) & test1["target"].isin(train_nodes)].reset_index(drop=True)

    for dfx in (train2, valid2, test2):
        if "__order__" in dfx.columns:
            pass 

    train2 = train2.drop(columns=["__order__"], errors="ignore")
    valid2 = valid2.drop(columns=["__order__"], errors="ignore")
    test2 = test2.drop(columns=["__order__"], errors="ignore")

    # Final remap: training nodes -> 0..N-1.
    train, _, n_train_nodes = _remap_by_nodes(train2, train_nodes) if len(train2) > 0 else (train2, {}, 0)
    valid, _, _ = _remap_by_nodes(valid2, train_nodes) if len(valid2) > 0 else (valid2, {}, 0)
    test, _, _ = _remap_by_nodes(test2, train_nodes) if len(test2) > 0 else (test2, {}, 0)

    meta = {
        "dataset": dataset,
        "type": "temporal_split_then_dedup",
        "seed": seed,
        "ratios": list(ratios),
        "policy": "time split (sorted by timestamp) -> per-split dedup keep-last by time order -> train GCC -> transductive filter -> remap by train nodes",
        "edges_total_after_full_gcc": int(m0),
        "edges_train_raw": int(len(train_raw)),
        "edges_valid_raw": int(len(valid_raw)),
        "edges_test_raw": int(len(test_raw)),
        "edges_train_after_dedup": int(len(train1)),
        "edges_valid_after_dedup": int(len(valid1)),
        "edges_test_after_dedup": int(len(test1)),
        "edges_train_after_train_gcc": int(len(train)),
        "edges_valid_after_transductive": int(len(valid)),
        "edges_test_after_transductive": int(len(test)),
        "n_train_nodes": int(n_train_nodes),
        "note": "val/test may include (u,v) pairs that also appear in train because dedup is applied per split.",
    }
    _write_split(dataset, train, valid, test, meta)


def main():
    temporal = ["bitcoinalpha", "bitcoinotc", "wiki-RfA", "wiki-Elec"]
    static = ["Slashdot", "Epinions"]


    for d in temporal:
        split_temporal_split_then_dedup(d)

    for d in static:
        split_static(d)


if __name__ == "__main__":
    import sys
    ds = sys.argv[1] if len(sys.argv) > 1 else None
    if ds is None or ds == "all":
        main()
    else:
        # decide split type by dataset name (same lists as in main)
        temporal = {"bitcoinalpha", "bitcoinotc", "wiki-RfA", "wiki-Elec"}
        if ds in temporal:
            split_temporal_split_then_dedup(ds)
        else:
            split_static(ds)

