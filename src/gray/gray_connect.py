#!/usr/bin/env python3
# src/gray/gray_connect.py
from __future__ import annotations

import argparse
import ast
import os
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from filelock import FileLock

from src.utils.paths import RESULTS_GRAY, DATA_SPLITS
from src.utils.seed import set_seed


MIN_COMM_SIZE = 30
SIGN_NEW = 1


def load_communities(path: Path, min_size: int = MIN_COMM_SIZE) -> Dict[int, List[int]]:
    df = pd.read_csv(path)
    out: Dict[int, List[int]] = {}
    for _, r in df.iterrows():
        cid = int(r["community_id"])
        try:
            nodes = ast.literal_eval(str(r["nodes"]))
        except Exception:
            nodes = []
        nodes = [int(x) for x in nodes]
        if len(nodes) >= min_size:
            out[cid] = nodes
    return out


def _count_intra_and_external(src: np.ndarray, trg: np.ndarray, nodes: np.ndarray) -> tuple[int, int]:
    s_in = np.isin(src, nodes)
    t_in = np.isin(trg, nodes)
    intra = int(np.sum(s_in & t_in))
    external = int(np.sum((s_in & ~t_in) | (t_in & ~s_in)))
    return intra, external


def _mode_reduce(a: int, b: int, mode: str) -> int:
    if mode == "min":
        return int(min(a, b))
    if mode == "avg":
        return int(round((a + b) / 2))
    if mode == "max":
        return int(max(a, b))
    raise ValueError(mode)


def _compute_global_targets(
    gray_mode: str,
    communities: Dict[int, List[int]],
    src_pos: np.ndarray,
    trg_pos: np.ndarray,
) -> tuple[int, int]:
    intra_list = []
    for nodes in communities.values():
        arr = np.asarray(nodes, dtype=int)
        intra_list.append(int(np.sum(np.isin(src_pos, arr) & np.isin(trg_pos, arr))))

    comm_ids = list(communities.keys())
    inter_list = []
    for i in range(len(comm_ids)):
        ci = np.asarray(communities[comm_ids[i]], dtype=int)
        for j in range(i + 1, len(comm_ids)):
            cj = np.asarray(communities[comm_ids[j]], dtype=int)
            inter_list.append(
                int(
                    np.sum(
                        (np.isin(src_pos, ci) & np.isin(trg_pos, cj))
                        | (np.isin(src_pos, cj) & np.isin(trg_pos, ci))
                    )
                )
            )

    def reduce_list(xs: list[int]) -> int:
        if not xs:
            return 0
        if gray_mode == "min":
            return int(min(xs))
        if gray_mode == "avg":
            return int(round(float(np.mean(xs))))
        return int(max(xs))

    return reduce_list(intra_list), reduce_list(inter_list)


def _compute_global_k_nodes(gray_mode: str, communities: Dict[int, List[int]]) -> int:
    sizes = [len(nodes) for nodes in communities.values()]
    if not sizes:
        return 0

    if gray_mode == "min":
        return int(min(sizes))
    if gray_mode == "avg":
        return int(round(float(np.mean(sizes))))
    if gray_mode == "max":
        return int(max(sizes))

    raise ValueError(gray_mode)


def _select_pairs(
    df_pairs: pd.DataFrame,
    minmax_th: float,
    max_degree: float,
    pair_selector: str,
    topk: int | None,
) -> list[pd.Series]:
    dmin = df_pairs["delta"].min()
    dmax = df_pairs["delta"].max()

    if pd.isna(dmin) or pd.isna(dmax) or dmax <= dmin:
        df_pairs = df_pairs.assign(delta_norm=1.0)
    else:
        df_pairs = df_pairs.assign(delta_norm=(df_pairs["delta"] - dmin) / (dmax - dmin))

    cand = df_pairs[df_pairs["delta_norm"] >= float(minmax_th)].copy()

    c1 = cand["community_1"].astype(int)
    c2 = cand["community_2"].astype(int)
    lo, hi = np.minimum(c1, c2), np.maximum(c1, c2)
    cand = cand.assign(_lo=lo, _hi=hi).sort_values(
        ["delta", "_lo", "_hi"], ascending=[False, True, True]
    )

    from collections import defaultdict
    deg_cap = defaultdict(int)
    D = int(1e18) if (np.isinf(max_degree) or max_degree >= 1e9) else int(max_degree)

    selected = []
    for _, row in cand.iterrows():
        a, b = int(row["community_1"]), int(row["community_2"])
        if deg_cap[a] < D and deg_cap[b] < D:
            selected.append(row)
            deg_cap[a] += 1
            deg_cap[b] += 1
            if pair_selector == "topk" and topk is not None and len(selected) >= int(topk):
                break
    return selected


def _read_gray_scores(model: str, dataset: str, seed: int, c1: int, c2: int, setting: str) -> pd.DataFrame | None:
    p1 = RESULTS_GRAY / model / dataset / f"seed{seed}" / "aug" / "gray_node" / f"pair_{c1}_{c2}.csv"
    p2 = RESULTS_GRAY / model / dataset / f"seed{seed}" / "aug" / "gray_node" / f"pair_{c2}_{c1}.csv"
    p = p1 if p1.exists() else (p2 if p2.exists() else None)
    if p is None:
        return None

    df = pd.read_csv(p)
    if "normalize" in df.columns:
        df = df[df["normalize"].astype(str) == str(setting)].copy()
    df = df.sort_values("score").drop_duplicates(subset=["node_id"], keep="first")
    return df


def _dir_selector(pair_selector: str, minmax_th: float, max_degree: float, topk: int | None) -> str:
    mm = f"mm{int(round(minmax_th * 100))}"
    deg = "degINF" if (np.isinf(max_degree) or max_degree >= 1e9) else f"deg{int(max_degree)}"
    if pair_selector == "threshold":
        return f"{pair_selector}_{mm}_{deg}"
    return f"{pair_selector}_{mm}_{deg}_topk{int(topk or 0)}"


def _dir_escale(edge_scale: float) -> str:
    v = int(round(edge_scale * 100))
    return f"escale_{v//100}p{v % 100:02d}"


def save_dir(
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="sgcn")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, required=True)

    ap.add_argument("--gray_mode", choices=["min", "avg", "max"], required=True)
    ap.add_argument("--scope", choices=["global", "local"], default="global")

    ap.add_argument("--pair_selector", choices=["threshold", "topk"], default="threshold")
    ap.add_argument("--minmax", type=float, default=0.0)
    ap.add_argument("--max_degree", type=float, default=1.0)
    ap.add_argument("--topk", type=int, default=None)

    ap.add_argument("--setting", choices=["none", "l2"], default="none")
    ap.add_argument("--edge_scale", type=float, default=1.0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    set_seed(int(args.seed))

    model = str(args.model)
    dataset = str(args.dataset)
    seed = int(args.seed)
    gray_mode = str(args.gray_mode)
    scope = str(args.scope)
    pair_selector = str(args.pair_selector)
    minmax_th = float(args.minmax)
    max_degree = float(args.max_degree)
    topk = args.topk
    setting = str(args.setting)
    edge_scale = float(args.edge_scale)

    out_dir = save_dir(
        model=model,
        dataset=dataset,
        seed=seed,
        gray_mode=gray_mode,
        scope=scope,
        pair_selector=pair_selector,
        minmax_th=minmax_th,
        max_degree=max_degree,
        topk=topk,
        setting=setting,
        edge_scale=edge_scale,
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    aug_out = out_dir / "train_edge_list_aug.csv"
    new_only_out = out_dir / "new_edges_only.csv"
    summary_out = out_dir / "gray_results.csv"
    lock = FileLock(str(out_dir / ".write.lock"))

    if aug_out.exists() and new_only_out.exists() and summary_out.exists() and not args.overwrite:
        print(f"⏭️ Already exists (skip): {out_dir}  (use --overwrite)")
        return

    train_path = DATA_SPLITS / dataset / "train_edge_list.csv"
    edge_df = pd.read_csv(train_path)[["source", "target", "weight"]].dropna().copy()
    edge_df["source"] = edge_df["source"].astype(int)
    edge_df["target"] = edge_df["target"].astype(int)
    edge_df["weight"] = edge_df["weight"].astype(int)

    num_nodes_seen = int(max(edge_df["source"].max(), edge_df["target"].max()) + 1)

    pos_mask = edge_df["weight"].to_numpy() > 0
    src_pos = edge_df["source"].to_numpy()[pos_mask]
    trg_pos = edge_df["target"].to_numpy()[pos_mask]

    comm_path = RESULTS_GRAY / model / dataset / f"seed{seed}" / "aug" / "kmeans_community.csv"
    communities = load_communities(comm_path, min_size=MIN_COMM_SIZE)
    num_communities = len(communities)

    pcs_path = RESULTS_GRAY / model / dataset / f"seed{seed}" / "aug" / "pcs" / "delta_pairs.csv"
    pcs_df = pd.read_csv(pcs_path)
    pcs_df = pcs_df[
        pcs_df["community_1"].astype(int).isin(communities.keys())
        & pcs_df["community_2"].astype(int).isin(communities.keys())
    ].copy()

    selected = _select_pairs(pcs_df, minmax_th, max_degree, pair_selector, topk)

    if len(selected) == 0:
        with lock:
            edge_df.to_csv(aug_out, index=False)
            pd.DataFrame(columns=["source", "target", "weight"]).to_csv(new_only_out, index=False)
            pd.DataFrame([{
                "model": model,
                "dataset": dataset,
                "seed": seed,
                "gray_mode": gray_mode,
                "scope": scope,
                "pair_selector": pair_selector,
                "minmax_th": minmax_th,
                "max_degree": max_degree,
                "topk": int(topk) if topk is not None else None,
                "setting": setting,
                "edge_scale": edge_scale,
                "selected_pairs": 0,
                "num_nodes_seen": num_nodes_seen,
                "num_communities": num_communities,
                "num_edges_added_total": 0,
            }]).to_csv(summary_out, index=False)
        print(f"✅ Saved empty artifacts: {out_dir}")
        return

    edge_set = set(zip(edge_df["source"], edge_df["target"])) | set(zip(edge_df["target"], edge_df["source"]))
    all_edges = list(zip(edge_df["source"], edge_df["target"], edge_df["weight"]))
    new_edges: list[tuple[int, int, int]] = []
    edge_set_local = set(edge_set)

    target_intra_g = target_inter_g = 0
    gray_k_global = 0
    if scope == "global":
        target_intra_g, target_inter_g = _compute_global_targets(gray_mode, communities, src_pos, trg_pos)
        gray_k_global = _compute_global_k_nodes(gray_mode, communities)

    summary_rows = []

    for row in selected:
        c1_id, c2_id = int(row["community_1"]), int(row["community_2"])
        c1_nodes, c2_nodes = communities[c1_id], communities[c2_id]

        gray_df = _read_gray_scores(model, dataset, seed, c1_id, c2_id, setting)
        if gray_df is None or gray_df.empty:
            continue

        if scope == "global":
            gray_k = max(0, int(gray_k_global))
            target_gg = int(round(max(0, int(target_intra_g)) * edge_scale))
            t_inter = int(round(max(0, int(target_inter_g)) * edge_scale))
            target_gC1 = t_inter // 2
            target_gC2 = t_inter - target_gC1
            intra1 = intra2 = ext1 = ext2 = inter_raw = None
        else:
            n1, n2 = len(c1_nodes), len(c2_nodes)
            if gray_mode == "min":
                gray_k = min(n1, n2)
            elif gray_mode == "avg":
                gray_k = int(round((n1 + n2) / 2))
            else:
                gray_k = max(n1, n2)
            gray_k = max(0, int(gray_k))

            C1 = np.asarray(c1_nodes, dtype=int)
            C2 = np.asarray(c2_nodes, dtype=int)
            intra1, ext1 = _count_intra_and_external(src_pos, trg_pos, C1)
            intra2, ext2 = _count_intra_and_external(src_pos, trg_pos, C2)
            target_gg_raw = _mode_reduce(intra1, intra2, gray_mode)

            c1_src = np.isin(src_pos, C1)
            c1_trg = np.isin(trg_pos, C1)
            c2_src = np.isin(src_pos, C2)
            c2_trg = np.isin(trg_pos, C2)
            inter_raw = int(np.sum((c1_src & c2_trg) | (c2_src & c1_trg)))

            target_gg = int(round(max(0, int(target_gg_raw)) * edge_scale))
            t_inter = int(round(max(0, int(inter_raw)) * edge_scale))
            target_gC1 = t_inter // 2
            target_gC2 = t_inter - target_gC1

        gray_nodes = gray_df.sort_values("score").head(gray_k)["node_id"].astype(int).tolist()

        gg_pairs = [(gray_nodes[i], gray_nodes[j]) for i in range(len(gray_nodes)) for j in range(i + 1, len(gray_nodes))]
        gg_pairs = [(u, v) for (u, v) in gg_pairs if (u, v) not in edge_set_local and (v, u) not in edge_set_local]
        random.shuffle(gg_pairs)

        gg_add = 0
        for u, v in gg_pairs[: min(int(target_gg), len(gg_pairs))]:
            all_edges.append((u, v, SIGN_NEW))
            new_edges.append((u, v, SIGN_NEW))
            edge_set_local.add((u, v))
            edge_set_local.add((v, u))
            gg_add += 1

        target_gC1 = int(max(0, target_gC1))
        target_gC2 = int(max(0, target_gC2))

        c1_pairs = [(g, t) for g in gray_nodes for t in c1_nodes if (g, t) not in edge_set_local and (t, g) not in edge_set_local]
        c2_pairs = [(g, t) for g in gray_nodes for t in c2_nodes if (g, t) not in edge_set_local and (t, g) not in edge_set_local]
        random.shuffle(c1_pairs)
        random.shuffle(c2_pairs)

        picked = c1_pairs[: min(target_gC1, len(c1_pairs))] + c2_pairs[: min(target_gC2, len(c2_pairs))]
        gc_add = 0
        for u, v in picked:
            all_edges.append((u, v, SIGN_NEW))
            new_edges.append((u, v, SIGN_NEW))
            edge_set_local.add((u, v))
            edge_set_local.add((v, u))
            gc_add += 1

        rec = {
            "model": model,
            "dataset": dataset,
            "seed": seed,
            "gray_mode": gray_mode,
            "scope": scope,
            "pair_selector": pair_selector,
            "minmax_th": minmax_th,
            "max_degree": max_degree,
            "topk": int(topk) if topk is not None else None,
            "setting": setting,
            "edge_scale": edge_scale,
            "community_1": c1_id,
            "community_2": c2_id,
            "num_gray_nodes": len(gray_nodes),
            "num_edges_added": gg_add + gc_add,
            "num_gray_gray_edges": gg_add,
            "num_gray_comm_edges": gc_add,
            "selected_pair_delta": float(row["delta"]),
        }
        if scope == "global":
            rec.update({
                "gray_k_global": int(gray_k_global),
                "avg_intra_pos_edges": int(target_intra_g),
                "avg_inter_pos_edges": int(target_inter_g),
            })
        else:
            rec.update({
                "intra_C1_pos": int(intra1),
                "intra_C2_pos": int(intra2),
                "external_C1_pos": int(ext1),
                "external_C2_pos": int(ext2),
                "inter_C1C2_pos_raw": int(inter_raw),
            })
        summary_rows.append(rec)

    tmp_aug = aug_out.with_suffix(".csv.tmp")
    tmp_new = new_only_out.with_suffix(".csv.tmp")
    tmp_sum = summary_out.with_suffix(".csv.tmp")

    with lock:
        pd.DataFrame(all_edges, columns=["source", "target", "weight"]).to_csv(tmp_aug, index=False)
        pd.DataFrame(new_edges, columns=["source", "target", "weight"]).to_csv(tmp_new, index=False)
        pd.DataFrame(summary_rows).to_csv(tmp_sum, index=False)

        os.replace(tmp_aug, aug_out)
        os.replace(tmp_new, new_only_out)
        os.replace(tmp_sum, summary_out)

    print(f"✅ Saved to: {out_dir}")
    print(f"   - selected_pairs: {len(selected)} | new_edges={len(new_edges)}")


if __name__ == "__main__":
    main()
