# src/train/train_sgcn.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd
import torch
from torch_geometric.utils import negative_sampling
from sklearn.metrics import accuracy_score, f1_score
from filelock import FileLock

from src.models.sgcn import SGCNForSignLinkClass
from src.utils.seed import set_seed
from src.utils.paths import DATA_SPLITS


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
    df["weight"] = df["weight"].astype(int)
    df = df[(df["weight"] != 0) & (df["source"] != df["target"])].reset_index(drop=True)
    return df


def _load_split_csv(dataset: str, split: str) -> pd.DataFrame:
    p = DATA_SPLITS / dataset / f"{split}_edge_list.csv"
    return _load_edge_csv(p)


def _remove_test_conflicts_undirected(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    s = df["source"].to_numpy(dtype=np.int64)
    t = df["target"].to_numpy(dtype=np.int64)
    u = np.minimum(s, t)
    v = np.maximum(s, t)
    sign = (df["weight"].to_numpy(dtype=np.int64) > 0).astype(np.int8)

    tmp = df.copy()
    tmp["_u"] = u
    tmp["_v"] = v
    tmp["_sign"] = sign

    nunique = tmp.groupby(["_u", "_v"])["_sign"].nunique()
    conflict_pairs = nunique[nunique > 1].index
    if len(conflict_pairs) > 0:
        conflict_set = set(conflict_pairs.tolist())
        keep = ~tmp.apply(lambda r: (int(r["_u"]), int(r["_v"])) in conflict_set, axis=1)
        tmp = tmp.loc[keep.values].copy()

    if not tmp.empty:
        tmp["__idx__"] = np.arange(len(tmp), dtype=np.int64)
        tmp = tmp.sort_values("__idx__", kind="mergesort").drop(columns="__idx__").copy()

    return tmp.drop(columns=["_u", "_v", "_sign"], errors="ignore").reset_index(drop=True)


def _split_pos_neg(df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
    if df.empty:
        empty = torch.empty((2, 0), dtype=torch.long)
        return empty, empty

    edge_index = torch.tensor(df[["source", "target"]].to_numpy(), dtype=torch.long).t()
    w = torch.tensor(df["weight"].to_numpy(), dtype=torch.long)
    pos = edge_index[:, w > 0]
    neg = edge_index[:, w < 0]
    return pos, neg


def _cache_path(cache_dir: Path, seed: int, split: str, tag: str | None = None) -> Path:
    # For valid/test, cache is fixed by seed (same evaluation as base).
    if split in ("valid", "test"):
        return cache_dir / f"seed{seed}_{split}_non.pt"
    # For train, cache is separated by tag (different non-edges per augmentation/config).
    tag = (tag or "base").strip().replace("/", "_")
    return cache_dir / f"seed{seed}_train_{tag}_non.pt"


def _make_undirected_forbidden(edge_index: torch.Tensor) -> torch.Tensor:
    """
    Given edge_index [2, E], return undirected forbidden edges [2, 2E]
    by adding flipped directions.
    """
    if edge_index.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.cat([edge_index, edge_index.flip(0)], dim=1).to(torch.long)


def _load_or_sample_non_edges(
    seed: int,
    split: str,
    tag: str | None,
    num_nodes: int,
    pos: torch.Tensor,
    neg: torch.Tensor,
    cache_dir: Path,
    # Extra forbidden edges (undirected), used to include train edges in the forbidden set for valid/test.
    extra_forbidden_ud: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Non-edge negative sampling.
    - train: forbid only split(pos+neg)
    - valid/test: forbid split(pos+neg) + train(pos+neg) (injected via extra_forbidden_ud)
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_path(cache_dir, seed, split, tag)

    lock = FileLock(str(cache_path) + ".lock")
    with lock:
        if cache_path.exists():
            return torch.load(cache_path, map_location="cpu").to(torch.long)

        if pos.numel() == 0 and neg.numel() == 0:
            non = torch.empty((2, 0), dtype=torch.long)
            torch.save(non, cache_path)
            return non

        full = torch.cat([pos, neg], dim=1)
        num_neg = int(full.size(1))  # sample the same number as (pos+neg)

        # Base forbidden set: observed edges in the current split (undirected).
        forbid_ud = _make_undirected_forbidden(full)

        # For valid/test, also include train edges in the forbidden set.
        if extra_forbidden_ud is not None and extra_forbidden_ud.numel() > 0:
            forbid_ud = torch.cat([forbid_ud, extra_forbidden_ud.to(torch.long)], dim=1)

        non = negative_sampling(
            edge_index=forbid_ud,
            num_nodes=num_nodes,
            num_neg_samples=num_neg,
        ).to(torch.long)

        torch.save(non.cpu(), cache_path)
        return non.cpu()


def train_sgcn_for_signlink_class(
    dataset: str,
    save_dir: str | Path,
    embedding_dim: int = 64,
    num_layers: int = 2,
    epochs: int = 400,
    lr: float = 0.01,
    patience: int = 10,
    seed: int = 42,
    num_communities=None,
    x: torch.Tensor | None = None,
    # Optional:
    train_edge_csv: str | Path | None = None,   # Provide an augmented train CSV if needed.
    train_tag: str | None = None,               # Tag to separate the train non-edge cache.
    **kwargs,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    set_seed(seed)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ----- load splits -----
    if train_edge_csv is None:
        train_df = _load_split_csv(dataset, "train")
    else:
        train_df = _load_edge_csv(Path(train_edge_csv))

    val_df = _load_split_csv(dataset, "valid")
    test_df = _load_split_csv(dataset, "test")
    val_df = _remove_test_conflicts_undirected(val_df)
    test_df = _remove_test_conflicts_undirected(test_df)

    if train_df.empty:
        raise ValueError(f"{dataset}: train split is empty")

    num_nodes = int(max(train_df["source"].max(), train_df["target"].max()) + 1)

    train_pos, train_neg = _split_pos_neg(train_df)
    val_pos, val_neg = _split_pos_neg(val_df)
    test_pos, test_neg = _split_pos_neg(test_df)

    cache_dir = DATA_SPLITS / dataset / "non_edges"

    # Train non-edges are separated by tag (can differ by augmentation).
    train_non = _load_or_sample_non_edges(seed, "train", train_tag, num_nodes, train_pos, train_neg, cache_dir)

    # Valid/test non-edges are fixed by seed, and train edges are included in the forbidden set.
    train_full = torch.cat([train_pos, train_neg], dim=1)
    train_forbidden_ud = _make_undirected_forbidden(train_full)

    val_non = _load_or_sample_non_edges(
        seed,
        "valid",
        None,
        num_nodes,
        val_pos,
        val_neg,
        cache_dir,
        extra_forbidden_ud=train_forbidden_ud,
    )
    test_non = _load_or_sample_non_edges(
        seed,
        "test",
        None,
        num_nodes,
        test_pos,
        test_neg,
        cache_dir,
        extra_forbidden_ud=train_forbidden_ud,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if x is None:
        x = torch.rand((num_nodes, 8), device=device)
    else:
        x = x.to(device)

    model = SGCNForSignLinkClass(
        in_channels=x.size(1),
        hidden_channels=embedding_dim,
        num_layers=num_layers,
        device=str(device),
        triplet_lambda=1.0,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_state = None
    best_z = None
    bad = 0
    ran_epochs = 0

    def _loss_for(split_pos, split_neg, split_non):
        z = model(x, split_pos.to(device), split_neg.to(device))
        loss = model.loss(z, split_pos.to(device), split_neg.to(device), split_non.to(device))
        return loss, z

    val_loss_history: list[float] = []

    for ep in range(1, epochs + 1):
        ran_epochs = ep
        model.train()
        opt.zero_grad()
        tr_loss, z_tr = _loss_for(train_pos, train_neg, train_non)
        tr_loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            va_loss, _ = _loss_for(val_pos, val_neg, val_non)

        cur_val = float(va_loss.item())
        val_loss_history.append(cur_val)

        print(f"[{dataset}] epoch={ep:03d} train={tr_loss.item():.4f} val={va_loss.item():.4f}")

        if cur_val < best_val_loss - 1e-6:
            best_val_loss = cur_val
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_z = z_tr.detach().cpu()
            bad = 0
        else:
            bad += 1
            if bad >= patience and cur_val > min(val_loss_history):
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()

    def _eval(split_pos, split_neg, split_non):
        if split_pos.numel() == 0 and split_neg.numel() == 0 and split_non.numel() == 0:
            return 0.0, 0.0

        with torch.no_grad():
            z = model(x, split_pos.to(device), split_neg.to(device))
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
        "edges_train_signed": int((train_pos.size(1) + train_neg.size(1))),
        "edges_valid_signed": int((val_pos.size(1) + val_neg.size(1))),
        "edges_test_signed_after_conflict": int((test_pos.size(1) + test_neg.size(1))),
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
