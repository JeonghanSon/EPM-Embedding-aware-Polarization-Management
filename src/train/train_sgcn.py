from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from filelock import FileLock
from sklearn.metrics import accuracy_score, f1_score

from src.models.sgcn import SGCNForSignLinkClass
from src.utils.paths import DATA_SPLITS
from src.utils.seed import set_seed


def _load_edge_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[["source", "target", "weight"]].copy()
    df = df.dropna(subset=["source", "target", "weight"]).copy()

    df["source"] = pd.to_numeric(df["source"], errors="coerce")
    df["target"] = pd.to_numeric(df["target"], errors="coerce")
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
    df = df.dropna(subset=["source", "target", "weight"]).copy()

    df["source"] = df["source"].astype(int)
    df["target"] = df["target"].astype(int)
    df["weight"] = np.sign(df["weight"].astype(float)).astype(int)

    df = df[(df["weight"] != 0) & (df["source"] != df["target"])].reset_index(drop=True)
    return df


def _load_split_csv(dataset: str, split: str) -> pd.DataFrame:
    return _load_edge_csv(DATA_SPLITS / dataset / f"{split}_edge_list.csv")


def _canonical_undirected_keep_last(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df[["source", "target", "weight"]].copy().reset_index(drop=True)

    out = df[["source", "target", "weight"]].copy()

    s = out["source"].to_numpy()
    t = out["target"].to_numpy()
    out["source"] = np.minimum(s, t)
    out["target"] = np.maximum(s, t)

    out["__idx__"] = np.arange(len(out), dtype=np.int64)
    out = out.sort_values("__idx__", kind="mergesort")
    out = out.drop_duplicates(subset=["source", "target"], keep="last")
    out = out.sort_values("__idx__", kind="mergesort").drop(columns=["__idx__"]).reset_index(drop=True)

    return out[["source", "target", "weight"]]


def _remove_test_conflicts_undirected(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    s = df["source"].to_numpy(dtype=np.int64)
    t = df["target"].to_numpy(dtype=np.int64)

    tmp = df.copy()
    tmp["_u"] = np.minimum(s, t)
    tmp["_v"] = np.maximum(s, t)
    tmp["_sign"] = (df["weight"].to_numpy(dtype=np.int64) > 0).astype(np.int8)

    nunique = tmp.groupby(["_u", "_v"])["_sign"].nunique()
    conflict_pairs = set(nunique[nunique > 1].index.tolist())

    if conflict_pairs:
        keep = ~tmp.apply(lambda r: (int(r["_u"]), int(r["_v"])) in conflict_pairs, axis=1)
        tmp = tmp.loc[keep.values].copy()

    return tmp.drop(columns=["_u", "_v", "_sign"], errors="ignore").reset_index(drop=True)


def _split_pos_neg(df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
    if df.empty:
        empty = torch.empty((2, 0), dtype=torch.long)
        return empty, empty

    edge_index = torch.tensor(df[["source", "target"]].to_numpy(), dtype=torch.long).t().contiguous()
    w = torch.tensor(df["weight"].to_numpy(), dtype=torch.long)

    pos = edge_index[:, w > 0]
    neg = edge_index[:, w < 0]
    return pos, neg


def _to_bidirectional(edge_index: torch.Tensor) -> torch.Tensor:
    if edge_index.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long)

    out = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    return torch.unique(out.t(), dim=0).t().contiguous()


def _pair_set_from_df(df: pd.DataFrame) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()

    if df.empty:
        return pairs

    for a, b in df[["source", "target"]].to_numpy(dtype=np.int64):
        u, v = (int(a), int(b)) if a < b else (int(b), int(a))
        if u != v:
            pairs.add((u, v))

    return pairs


def _cache_path(cache_dir: Path, seed: int, split: str, tag: str | None = None) -> Path:
    if split in ("valid", "test"):
        return cache_dir / f"seed{seed}_{split}_non.pt"

    tag = (tag or "base").strip().replace("/", "_")
    return cache_dir / f"seed{seed}_train_{tag}_non.pt"


def _sample_undirected_non_edges(
    seed: int,
    split: str,
    num_nodes: int,
    num_samples: int,
    forbidden: set[tuple[int, int]],
) -> torch.Tensor:
    if num_samples == 0:
        return torch.empty((2, 0), dtype=torch.long)

    offset = {"train": 0, "valid": 10_000_000, "test": 20_000_000}.get(split, 30_000_000)
    rng = np.random.default_rng(seed + offset)

    sampled: set[tuple[int, int]] = set()
    max_trials = max(100_000, num_samples * 100)
    trials = 0

    while len(sampled) < num_samples and trials < max_trials:
        a = int(rng.integers(0, num_nodes))
        b = int(rng.integers(0, num_nodes))
        trials += 1

        if a == b:
            continue

        u, v = (a, b) if a < b else (b, a)

        if (u, v) in forbidden or (u, v) in sampled:
            continue

        sampled.add((u, v))

    if len(sampled) < num_samples:
        raise RuntimeError(
            f"Failed to sample non-edges: requested={num_samples}, sampled={len(sampled)}"
        )

    arr = np.array(sorted(sampled), dtype=np.int64)
    return torch.tensor(arr.T, dtype=torch.long).contiguous()


def _load_or_sample_non_edges(
    seed: int,
    split: str,
    tag: str | None,
    num_nodes: int,
    num_samples: int,
    forbidden: set[tuple[int, int]],
    cache_dir: Path,
) -> torch.Tensor:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_path(cache_dir, seed, split, tag)

    lock = FileLock(str(cache_path) + ".lock")
    with lock:
        if cache_path.exists():
            return torch.load(cache_path, map_location="cpu").to(torch.long)

        non = _sample_undirected_non_edges(
            seed=seed,
            split=split,
            num_nodes=num_nodes,
            num_samples=num_samples,
            forbidden=forbidden,
        )

        torch.save(non.cpu(), cache_path)
        return non.cpu()


def _build_model(
    model_name: str,
    device: torch.device,
    x: torch.Tensor,
    embedding_dim: int,
    num_layers: int,
    kwargs: dict[str, Any],
) -> SGCNForSignLinkClass:
    if model_name.lower() != "sgcn":
        raise ValueError(f"Unsupported model_name: {model_name}")

    return SGCNForSignLinkClass(
        in_channels=x.size(1),
        hidden_channels=embedding_dim,
        num_layers=num_layers,
        device=str(device),
        triplet_lambda=float(kwargs.get("triplet_lambda", 1.0)),
    ).to(device)


def train_sgcn_for_signlink_class(
    dataset: str,
    save_dir: str | Path,
    embedding_dim: int = 64,
    num_layers: int = 2,
    epochs: int = 400,
    lr: float = 0.01,
    patience: int = 10,
    seed: int = 0,
    num_communities=None,
    x: torch.Tensor | None = None,
    train_edge_csv: str | Path | None = None,
    train_tag: str | None = None,
    model_name: str = "sgcn",
    **kwargs,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    set_seed(seed)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    base_train_df = _canonical_undirected_keep_last(_load_split_csv(dataset, "train"))
    val_df = _canonical_undirected_keep_last(
        _remove_test_conflicts_undirected(_load_split_csv(dataset, "valid"))
    )
    test_df = _canonical_undirected_keep_last(
        _remove_test_conflicts_undirected(_load_split_csv(dataset, "test"))
    )

    if train_edge_csv is None:
        train_df = base_train_df
    else:
        train_df = _canonical_undirected_keep_last(_load_edge_csv(Path(train_edge_csv)))

    if train_df.empty:
        raise ValueError(f"{dataset}: train split is empty")

    num_nodes = int(max(train_df["source"].max(), train_df["target"].max()) + 1)

    train_pos, train_neg = _split_pos_neg(train_df)
    val_pos, val_neg = _split_pos_neg(val_df)
    test_pos, test_neg = _split_pos_neg(test_df)

    train_pos_msg = _to_bidirectional(train_pos)
    train_neg_msg = _to_bidirectional(train_neg)

    observed_pairs = (
        _pair_set_from_df(base_train_df)
        | _pair_set_from_df(val_df)
        | _pair_set_from_df(test_df)
    )

    train_forbidden = observed_pairs | _pair_set_from_df(train_df)

    cache_dir = DATA_SPLITS / dataset / "non_edges"

    train_non = _load_or_sample_non_edges(
        seed=seed,
        split="train",
        tag=train_tag,
        num_nodes=num_nodes,
        num_samples=int(train_pos.size(1) + train_neg.size(1)),
        forbidden=train_forbidden,
        cache_dir=cache_dir,
    )

    val_non = _load_or_sample_non_edges(
        seed=seed,
        split="valid",
        tag=None,
        num_nodes=num_nodes,
        num_samples=int(val_pos.size(1) + val_neg.size(1)),
        forbidden=observed_pairs,
        cache_dir=cache_dir,
    )

    test_non = _load_or_sample_non_edges(
        seed=seed,
        split="test",
        tag=None,
        num_nodes=num_nodes,
        num_samples=int(test_pos.size(1) + test_neg.size(1)),
        forbidden=observed_pairs,
        cache_dir=cache_dir,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if x is None:
        x = torch.rand((num_nodes, 8), device=device)
    else:
        x = x.to(device)

    model = _build_model(
        model_name=model_name,
        device=device,
        x=x,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        kwargs=kwargs,
    )

    model_x = x.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def _encode_train_graph() -> torch.Tensor:
        return model(
            model_x,
            train_pos_msg.to(device),
            train_neg_msg.to(device),
        )

    def _train_loss() -> Tuple[torch.Tensor, torch.Tensor]:
        z = _encode_train_graph()
        loss = model.loss(
            z,
            train_pos.to(device),
            train_neg.to(device),
            train_non.to(device),
        )
        return loss, z

    def _valid_loss() -> Tuple[torch.Tensor, torch.Tensor]:
        z = _encode_train_graph()
        loss = model.nll_loss(
            z,
            val_pos.to(device),
            val_neg.to(device),
            val_non.to(device),
        )
        return loss, z

    best_val_loss = float("inf")
    best_state = None
    bad = 0
    ran_epochs = 0
    val_loss_history: list[float] = []

    for ep in range(1, epochs + 1):
        ran_epochs = ep

        model.train()
        opt.zero_grad()
        tr_loss, _ = _train_loss()
        tr_loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            va_loss, _ = _valid_loss()

        cur_val = float(va_loss.item())
        val_loss_history.append(cur_val)

        print(
            f"[{dataset}|{model_name}] epoch={ep:03d} "
            f"train={tr_loss.item():.4f} val={va_loss.item():.4f}"
        )

        if cur_val < best_val_loss - 1e-6:
            best_val_loss = cur_val
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience and cur_val > min(val_loss_history):
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    model.eval()
    with torch.no_grad():
        best_z = _encode_train_graph().detach().cpu()

    def _eval(split_pos: torch.Tensor, split_neg: torch.Tensor, split_non: torch.Tensor):
        if split_pos.numel() == 0 and split_neg.numel() == 0 and split_non.numel() == 0:
            return 0.0, 0.0

        with torch.no_grad():
            z = best_z.to(device)
            edges = torch.cat([split_pos, split_neg, split_non], dim=1).to(device)
            pred = model.discriminate_z(z, edges).argmax(dim=1).cpu().numpy()

        y = torch.cat(
            [
                torch.zeros(split_pos.size(1), dtype=torch.long),
                torch.ones(split_neg.size(1), dtype=torch.long),
                torch.full((split_non.size(1),), 2, dtype=torch.long),
            ]
        ).numpy()

        return float(accuracy_score(y, pred)), float(f1_score(y, pred, average="macro"))

    val_acc, val_f1 = _eval(val_pos, val_neg, val_non)
    test_acc, test_f1 = _eval(test_pos, test_neg, test_non)

    z_path = save_dir / "z.pt"
    model_path = save_dir / "model.pt"
    metrics_path = save_dir / "metrics.json"

    torch.save(best_z, z_path)
    torch.save(best_state, model_path)

    metrics = {
        "model": model_name,
        "dataset": dataset,
        "seed": seed,
        "embedding_dim": embedding_dim,
        "num_layers": num_layers,
        "lr": lr,
        "epochs_ran": ran_epochs,
        "patience": patience,
        "best_val_loss": best_val_loss,
        "val_accuracy": val_acc,
        "val_f1": val_f1,
        "test_accuracy": test_acc,
        "test_f1": test_f1,
        "num_nodes": num_nodes,
        "edges_train_signed": int(train_pos.size(1) + train_neg.size(1)),
        "edges_valid_signed": int(val_pos.size(1) + val_neg.size(1)),
        "edges_test_signed_after_conflict": int(test_pos.size(1) + test_neg.size(1)),
        "non_train": int(train_non.size(1)),
        "non_valid": int(val_non.size(1)),
        "non_test": int(test_non.size(1)),
        "test_conflict_filtered": True,
        "non_edge_cache_dir": str(cache_dir),
        "train_non_tag": (train_tag or "base"),
        "train_edge_csv": str(train_edge_csv) if train_edge_csv is not None else None,
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    artifacts = {
        "z_path": str(z_path),
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
    }

    return metrics, artifacts
