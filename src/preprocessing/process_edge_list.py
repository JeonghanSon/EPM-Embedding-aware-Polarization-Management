# src/preprocessing/process_edge_list.py
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import pandas as pd

from src.utils.paths import DATA_RAW, DATA_INTERIM


def _save(df: pd.DataFrame, dataset_name: str) -> Path:
    out_dir = DATA_INTERIM / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "edge_list.csv"
    df.to_csv(out_path, index=False)
    print(f"✅ Saved {dataset_name} -> {out_path} (edges={len(df)})")
    return out_path


def process_bitcoin(dataset_name: str) -> Path:
    """
    SNAP raw files may come as .csv or .txt depending on the dataset mirror/version.
    We try both to make preprocessing more robust.
    """
    cand = [
        DATA_RAW / f"soc-sign-{dataset_name}.csv",
        DATA_RAW / f"soc-sign-{dataset_name}.txt",
    ]
    inp = next((p for p in cand if p.exists()), None)
    if inp is None:
        raise FileNotFoundError(f"Missing raw file for {dataset_name}. Tried: {cand}")

    df = pd.read_csv(inp, header=None, names=["source", "target", "weight", "timestamp"])
    df = df.dropna(subset=["source", "target", "weight", "timestamp"])
    df = df[df["weight"] != 0]
    df = df.sort_values("timestamp").reset_index(drop=True)
    return _save(df, dataset_name)



def process_wiki_rfa() -> Path:
    dataset_name = "wiki-RfA"
    inp = DATA_RAW / "wiki-RfA.txt"

    def parse_dat(dat_str: str):
        dat_str = (dat_str or "").strip()
        if not dat_str:
            return None
        try:
            return int(datetime.strptime(dat_str, "%H:%M, %d %B %Y").timestamp())
        except Exception:
            return None

    edges, cur = [], {}
    last_ts = None

    def flush():
        nonlocal last_ts, cur, edges
        if not cur:
            return
        if not all(k in cur for k in ("source", "target", "weight")):
            cur = {}
            return
        if int(cur["weight"]) == 0:
            cur = {}
            return

        ts_cand = parse_dat(cur.get("dat_raw", ""))
        if ts_cand is None:
            if last_ts is None:
                y = cur.get("year", None)
                ts = int(datetime(int(y), 1, 1).timestamp()) if y is not None else 0
            else:
                ts = int(last_ts + 1)
        else:
            ts = int(ts_cand) if last_ts is None or ts_cand > last_ts else int(last_ts + 1)

        last_ts = ts
        edges.append(
            {
                "source": cur["source"],
                "target": cur["target"],
                "weight": int(cur["weight"]),
                "timestamp": int(ts),
            }
        )
        cur = {}

    with open(inp, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("SRC:"):
                cur["source"] = line[4:].strip()
            elif line.startswith("TGT:"):
                cur["target"] = line[4:].strip()
            elif line.startswith("VOT:"):
                try:
                    cur["weight"] = int(line[4:].strip())
                except Exception:
                    cur["weight"] = 0
            elif line.startswith("YEA:"):
                try:
                    cur["year"] = int(line[4:].strip())
                except Exception:
                    cur["year"] = None
            elif line.startswith("DAT:"):
                cur["dat_raw"] = line[4:].strip()
            elif line == "":
                flush()

    flush()

    df = pd.DataFrame(edges)
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    return _save(df, dataset_name)


def process_wiki_elec() -> Path:
    dataset_name = "wiki-Elec"
    inp = DATA_RAW / "wikiElec.ElecBs3.txt"

    edges = []
    current_target = None

    with open(inp, "r", encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if line.startswith("U\t"):
                parts = line.split("\t")
                if len(parts) >= 2:
                    current_target = parts[1]
            elif line.startswith("V\t") and current_target is not None:
                parts = line.split("\t")
                if len(parts) >= 4:
                    try:
                        weight = int(parts[1])
                    except Exception:
                        continue
                    source = parts[2]
                    ts = parts[3]
                    if weight != 0 and source and ts:
                        edges.append(
                            {"source": source, "target": current_target, "weight": weight, "timestamp": ts}
                        )

    df = pd.DataFrame(edges)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return _save(df, dataset_name)


def process_snap_static(dataset_name: str, raw_filename: str) -> Path:
    inp = DATA_RAW / raw_filename
    df = pd.read_csv(
        inp,
        comment="#",
        header=None,
        sep=r"\s+",
        names=["source", "target", "weight"],
        engine="python",
    )
    df = df.dropna(subset=["source", "target", "weight"])
    df["source"] = df["source"].astype(int)
    df["target"] = df["target"].astype(int)
    df["weight"] = df["weight"].astype(int)
    df = df[(df["weight"] != 0) & (df["source"] != df["target"])]

    s = df["source"].to_numpy()
    t = df["target"].to_numpy()
    u, v = s.copy(), t.copy()
    m = u > v
    u[m], v[m] = v[m], u[m]
    df["source"], df["target"] = u, v

    return _save(df, dataset_name)


def run_all():
    process_bitcoin("bitcoinalpha")
    process_bitcoin("bitcoinotc")
    process_wiki_rfa()
    process_wiki_elec()
    process_snap_static("Slashdot", "soc-sign-Slashdot090221.txt")
    process_snap_static("Epinions", "soc-sign-epinions.txt")


def run_one(dataset: str):
    if dataset == "bitcoinalpha":
        return process_bitcoin("bitcoinalpha")
    if dataset == "bitcoinotc":
        return process_bitcoin("bitcoinotc")
    if dataset == "wiki-RfA":
        return process_wiki_rfa()
    if dataset == "wiki-Elec":
        return process_wiki_elec()
    if dataset == "Slashdot":
        return process_snap_static("Slashdot", "soc-sign-Slashdot090221.txt")
    if dataset == "Epinions":
        return process_snap_static("Epinions", "soc-sign-epinions.txt")
    raise ValueError(f"Unknown dataset: {dataset}")


if __name__ == "__main__":
    import sys
    ds = sys.argv[1] if len(sys.argv) > 1 else None
    if ds is None or ds == "all":
        run_all()
    else:
        run_one(ds)
