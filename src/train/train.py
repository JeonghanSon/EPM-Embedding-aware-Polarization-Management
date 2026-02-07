import argparse
import json
import shutil
from pathlib import Path
from filelock import FileLock

import pandas as pd

from src.utils.paths import RESULTS_BASE, DATA_META, DATA_SPLITS
from src.utils.seed import set_seed
from src.train.train_sgcn import train_sgcn_for_signlink_class


RESULTS_CSV = RESULTS_BASE / "embedding_results.csv"
BEST_CSV = RESULTS_BASE / "best_embeddings.csv"


def extract_num_communities(meta_path: Path):
    if not meta_path.exists():
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return meta.get("mode", meta.get("avg", None))


def run_grid(
    dataset: str,
    seed: int,
    embedding_dims,
    num_layers_list,
    learning_rates,
    epochs: int,
):
    split_dir = DATA_SPLITS / dataset
    if not (split_dir / "train_edge_list.csv").exists():
        print(f"Missing splits for {dataset}: {split_dir}")
        return [], None

    num_comm = extract_num_communities(DATA_META / dataset / "num_communities.json")

    base_dir = RESULTS_BASE / dataset / f"seed{seed}"
    base_dir.mkdir(parents=True, exist_ok=True)

    tmp_root = base_dir / "_tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)

    best_row = None
    best_val_f1 = -1.0
    rows = []

    for dim in embedding_dims:
        for L in num_layers_list:
            for lr in learning_rates:
                run_name = f"dim{dim}_L{L}_lr{lr}"
                run_dir = tmp_root / run_name
                if run_dir.exists():
                    shutil.rmtree(run_dir)
                run_dir.mkdir(parents=True, exist_ok=True)

                print(f"[{dataset}] {run_name} seed={seed}")

                metrics, artifacts = train_sgcn_for_signlink_class(
                    dataset=dataset,
                    save_dir=run_dir,
                    embedding_dim=dim,
                    num_layers=L,
                    epochs=epochs,
                    lr=lr,
                    seed=seed,
                    num_communities=num_comm,
                )

                row = {
                    "dataset": dataset,
                    "seed": seed,
                    "num_communities": num_comm,
                    "embedding_dim": dim,
                    "num_layers": L,
                    "lr": lr,
                    "val_accuracy": metrics.get("val_accuracy", None),
                    "val_f1": metrics.get("val_f1", None),
                    "test_accuracy": metrics.get("test_accuracy", None),
                    "test_f1": metrics.get("test_f1", None),
                }
                rows.append(row)

                val_f1 = row["val_f1"]
                if val_f1 is not None and float(val_f1) > best_val_f1:
                    best_val_f1 = float(val_f1)
                    best_row = row

                    z_src = Path(artifacts.get("z_path", ""))
                    m_src = Path(artifacts.get("model_path", ""))
                    j_src = Path(artifacts.get("metrics_path", ""))

                    if z_src.exists():
                        shutil.copyfile(z_src, base_dir / "best_z.pt")
                    if m_src.exists():
                        shutil.copyfile(m_src, base_dir / "best_model.pt")
                    if j_src.exists():
                        shutil.copyfile(j_src, base_dir / "best_metrics.json")

                    best_cfg = {
                        "dataset": dataset,
                        "seed": seed,
                        "num_communities": num_comm,
                        "embedding_dim": dim,
                        "num_layers": L,
                        "lr": lr,
                        "val_f1": row["val_f1"],
                        "val_accuracy": row["val_accuracy"],
                        "test_f1": row["test_f1"],
                        "test_accuracy": row["test_accuracy"],
                    }
                    with open(base_dir / "best_config.json", "w", encoding="utf-8") as f:
                        json.dump(best_cfg, f, indent=2)

                shutil.rmtree(run_dir, ignore_errors=True)

    shutil.rmtree(tmp_root, ignore_errors=True)
    return rows, best_row


def _upsert_csv(path: Path, new_df: pd.DataFrame, key_cols: list[str], replace_mask: pd.Series | None = None):
    if path.exists():
        prev = pd.read_csv(path)
        if replace_mask is None:
            merged = pd.concat([prev, new_df], ignore_index=True)
        else:
            prev = prev[~replace_mask]
            merged = pd.concat([prev, new_df], ignore_index=True)
    else:
        merged = new_df
    merged.to_csv(path, index=False)


def update_results_csv(rows: list[dict], seed: int, datasets: list[str]):
    if not rows:
        return
    df = pd.DataFrame(rows)
    cols = [
        "dataset", "seed", "num_communities",
        "embedding_dim", "num_layers", "lr",
        "val_accuracy", "val_f1", "test_accuracy", "test_f1",
    ]
    df = df[cols]

    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)

    if RESULTS_CSV.exists():
        prev = pd.read_csv(RESULTS_CSV)
        mask = (prev["seed"] == seed) & (prev["dataset"].isin(datasets))
        prev = prev[~mask]
        merged = pd.concat([prev, df], ignore_index=True)
    else:
        merged = df

    merged.to_csv(RESULTS_CSV, index=False)
    print(f"Saved: {RESULTS_CSV}")


def update_best_csv(best_rows: list[dict], seed: int, datasets: list[str]):
    best_rows = [r for r in best_rows if r is not None]
    if not best_rows:
        return

    df = pd.DataFrame(best_rows)
    cols = [
        "dataset", "seed", "num_communities",
        "embedding_dim", "num_layers", "lr",
        "val_accuracy", "val_f1", "test_accuracy", "test_f1",
    ]
    df = df[cols]

    BEST_CSV.parent.mkdir(parents=True, exist_ok=True)

    if BEST_CSV.exists():
        prev = pd.read_csv(BEST_CSV)
        mask = (prev["seed"] == seed) & (prev["dataset"].isin(datasets))
        prev = prev[~mask]
        merged = pd.concat([prev, df], ignore_index=True)
    else:
        merged = df

    merged.to_csv(BEST_CSV, index=False)
    print(f"Saved: {BEST_CSV}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--datasets", nargs="+", default=["bitcoinalpha"])
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--embedding_dims", nargs="+", type=int, default=[32, 64, 128])
    p.add_argument("--num_layers", nargs="+", type=int, default=[2, 3, 4])
    p.add_argument("--lrs", nargs="+", type=float, default=[0.05, 0.01, 0.005, 0.001, 0.0005])
    args = p.parse_args()

    set_seed(args.seed)

    all_rows = []
    best_rows = []
    processed = []

    for dataset in args.datasets:
        rows, best = run_grid(
            dataset=dataset,
            seed=args.seed,
            embedding_dims=args.embedding_dims,
            num_layers_list=args.num_layers,
            learning_rates=args.lrs,
            epochs=args.epochs,
        )
        if rows:
            all_rows.extend(rows)
            processed.append(dataset)
        if best is not None:
            best_rows.append(best)
            print(f"Best [{dataset}] val_f1={best['val_f1']}")

    lock_path = RESULTS_BASE / ".csv.lock"
    lock = FileLock(str(lock_path))

    with lock:
        update_results_csv(all_rows, seed=args.seed, datasets=processed)
        update_best_csv(best_rows, seed=args.seed, datasets=processed)


if __name__ == "__main__":
    main()
