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

    df["source"] = pd.to_numeric(df["source"], errors="coerce")
    df["target"] = pd.to_numeric(df["target"], errors="coerce")
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
    df = df.dropna(subset=["source", "target", "weight"]).copy()

    df["source"] = df["source"].astype(int)
    df["target"] = df["target"].astype(int)
    df["weight"] = np.sign(df["weight"]).astype(int)

    df = df[(df["weight"] != 0) & (df["source"] != df["target"])].reset_index(drop=True)
    return df


def _canonical_undirected_keep_last(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df[["source", "target", "weight"]].copy().reset_index(drop=True)

    out = df.copy()

    s = out["source"].to_numpy()
    t = out["target"].to_numpy()
    out["source"] = np.minimum(s, t)
    out["target"] = np.maximum(s, t)

    out["__idx__"] = np.arange(len(out), dtype=np.int64)
    out = out.sort_values("__idx__", kind="mergesort")
    out = out.drop_duplicates(subset=["source", "target"], keep="last")
    out = out.sort_values("__idx__", kind="mergesort").drop(columns=["__idx__"]).reset_index(drop=True)

    return out[["source", "target", "weight"]]


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


def _remap_by_nodes(df: pd.DataFrame, nodes: np.ndarray) -> tuple[pd.DataFrame, dict[int, int], int]:
    nodes = np.asarray(nodes, dtype=np.int64)
    nodes_sorted = np.sort(nodes)
    node_map = {int(u): i for i, u in enumerate(nodes_sorted)}

    out = df.copy()
    out["source"] = out["source"].map(node_map)
    out["target"] = out["target"].map(node_map)
    out = out.dropna(subset=["source", "target"]).copy()

    out["source"] = out["source"].astype(int)
    out["target"] = out["target"].astype(int)
    out["weight"] = out["weight"].astype(int)

    return out[["source", "target", "weight"]].reset_index(drop=True), node_map, len(nodes_sorted)


def _write_split(dataset: str, train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame, meta: dict) -> None:
    out_dir = DATA_SPLITS / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    train.to_csv(out_dir / "train_edge_list.csv", index=False)
    valid.to_csv(out_dir / "valid_edge_list.csv", index=False)
    test.to_csv(out_dir / "test_edge_list.csv", index=False)

    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(
        f"{dataset} saved: "
        f"train={len(train)}, valid={len(valid)}, test={len(test)}, nodes(train)={meta['n_train_nodes']}"
    )


def _spanning_tree_edge_indices(df: pd.DataFrame, n_nodes: int) -> np.ndarray:
    src = df["source"].to_numpy()
    tgt = df["target"].to_numpy()

    adj = [[] for _ in range(n_nodes)]
    edge_id = {}

    for i, (a, b) in enumerate(zip(src, tgt)):
        a = int(a)
        b = int(b)
        adj[a].append(b)
        adj[b].append(a)
        edge_id[(min(a, b), max(a, b))] = i

    visited = np.zeros(n_nodes, dtype=bool)
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

    return np.asarray(tree_edges, dtype=np.int64)


def split_static_snapshot(
    dataset: str,
    seed: int = SEED,
    ratios: tuple[float, float, float] = RATIOS,
) -> None:
    inp = DATA_BUILD / dataset / "edge_list.csv"
    df_raw = pd.read_csv(inp)

    df = _clean_common(df_raw)
    n_edges_after_clean = int(len(df))

    # For temporal datasets, build_edge_list.py already sorted rows by timestamp.
    # Therefore, keep-last here means the final sign in the full static snapshot.
    df = _canonical_undirected_keep_last(df)
    n_edges_after_dedup = int(len(df))

    gcc_nodes = _gcc_nodes_from_edges(df["source"].to_numpy(), df["target"].to_numpy())
    df = df[df["source"].isin(gcc_nodes) & df["target"].isin(gcc_nodes)].reset_index(drop=True)
    n_edges_after_lcc = int(len(df))

    df, _, n_nodes = _remap_by_nodes(df, np.unique(df[["source", "target"]].to_numpy()))

    tree_edges = _spanning_tree_edge_indices(df, n_nodes)

    m = len(df)
    n_train = int(round(m * ratios[0]))
    n_valid = int(round(m * ratios[1]))
    n_test = m - n_train - n_valid

    if n_train < len(tree_edges):
        raise RuntimeError(
            f"{dataset}: train split too small to contain spanning tree "
            f"(train={n_train}, tree={len(tree_edges)})"
        )

    rng = np.random.default_rng(seed)

    all_idx = np.arange(m, dtype=np.int64)
    tree_set = set(tree_edges.tolist())
    rest = np.asarray([i for i in all_idx if i not in tree_set], dtype=np.int64)
    rng.shuffle(rest)

    need = n_train - len(tree_edges)

    train_idx = np.concatenate([tree_edges, rest[:need]])
    remain = rest[need:]

    valid_idx = remain[:n_valid]
    test_idx = remain[n_valid:n_valid + n_test]

    train = df.iloc[train_idx].reset_index(drop=True)
    valid = df.iloc[valid_idx].reset_index(drop=True)
    test = df.iloc[test_idx].reset_index(drop=True)

    train_nodes = np.unique(train[["source", "target"]].to_numpy())

    valid_before = len(valid)
    test_before = len(test)

    valid = valid[
        valid["source"].isin(train_nodes) & valid["target"].isin(train_nodes)
    ].reset_index(drop=True)
    test = test[
        test["source"].isin(train_nodes) & test["target"].isin(train_nodes)
    ].reset_index(drop=True)

    train, _, n_train_nodes = _remap_by_nodes(train, train_nodes)
    valid, _, _ = _remap_by_nodes(valid, train_nodes)
    test, _, _ = _remap_by_nodes(test, train_nodes)

    meta = {
        "dataset": dataset,
        "type": "static_final_snapshot_random_split",
        "seed": int(seed),
        "ratios": list(ratios),
        "policy": (
            "clean signed edges -> undirected final snapshot keep-last -> full LCC "
            "-> random 70/15/15 split with train spanning tree -> transductive filter"
        ),
        "edges_raw": int(len(df_raw)),
        "edges_after_clean": n_edges_after_clean,
        "edges_after_final_dedup": n_edges_after_dedup,
        "edges_after_lcc": n_edges_after_lcc,
        "edges_train": int(len(train)),
        "edges_valid": int(len(valid)),
        "edges_test": int(len(test)),
        "edges_valid_before_transductive": int(valid_before),
        "edges_test_before_transductive": int(test_before),
        "n_train_nodes": int(n_train_nodes),
        "train_contains_spanning_tree": True,
        "valid_dropped_by_transductive": int(valid_before - len(valid)),
        "test_dropped_by_transductive": int(test_before - len(test)),
    }

    _write_split(dataset, train, valid, test, meta)


def main() -> None:
    datasets = [
        "bitcoinalpha",
        "bitcoinotc",
        "wiki-RfA",
        "wiki-Elec",
        "Slashdot",
        "Epinions",
    ]

    for dataset in datasets:
        split_static_snapshot(dataset)


if __name__ == "__main__":
    import sys

    ds = sys.argv[1] if len(sys.argv) > 1 else "all"

    if ds == "all":
        main()
    else:
        split_static_snapshot(ds)
