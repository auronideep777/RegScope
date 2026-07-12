"""
Ingest the Agarwal/LegNet processed MPRA tables (K562 / HepG2 / WTC11) into the
same schema as activity_seq.csv, so they can be added as extra multi-task heads.

Input : human_legnet-main/datasets/original/<CELL>_averaged.tsv
        columns: seq_id, seq, mean_value, fold, rev
Output: promoter_seq_<CELL>.csv
        columns: insert_name, chrom, start, end, assay, log2_ratio, is_active,
                 is_silencer, sequence

Notes
- rev==1 rows are reverse-complement duplicates -> dropped (we RC-augment anyway).
- No genomic coordinates in this table, so `chrom` is used purely as a SPLIT LABEL:
  the LegNet CV fold is mapped to chr7 (test) / chr8 (val) / chr1 (train) so the
  existing chromosome-held-out splitter routes these rows correctly. It is NOT a
  real chromosome.
- is_active / is_silencer are left blank (masked) — these libraries don't carry
  those labels; only the activity-regression head is supervised for them.

    python src/prepare_promoter.py \
        --tsv human_legnet-main/datasets/original/WTC11_averaged.tsv \
        --assay WTC11 --outdir data
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True, help="path to <CELL>_averaged.tsv")
    ap.add_argument("--assay", required=True, help="assay/cell-type name, e.g. WTC11")
    ap.add_argument("--test-fold", type=int, default=10)
    ap.add_argument("--val-fold", type=int, default=9)
    ap.add_argument("--outdir", default="data")
    args = ap.parse_args()

    d = pd.read_csv(args.tsv, sep="\t")
    need = {"seq", "mean_value"}
    if not need.issubset(d.columns):
        raise SystemExit(f"expected columns {need}, got {list(d.columns)}")
    if "rev" in d.columns:
        d = d[d["rev"] == 0]                       # drop RC-duplicate rows
    id_col = "seq_id" if "seq_id" in d.columns else None
    fold = pd.to_numeric(d.get("fold", pd.Series([0] * len(d))), errors="coerce").fillna(0).astype(int)

    def split_label(f):
        if f == args.test_fold:
            return "chr7"       # -> held-out test in chrom_split
        if f == args.val_fold:
            return "chr8"       # -> validation
        return "chr1"           # -> train

    out = pd.DataFrame({
        "insert_name": (d[id_col].astype(str) if id_col else [f"{args.assay}_{i}" for i in range(len(d))]),
        "chrom": [split_label(f) for f in fold],
        "start": 0, "end": d["seq"].astype(str).str.len().values,
        "assay": args.assay,
        "log2_ratio": pd.to_numeric(d["mean_value"], errors="coerce").values,
        "is_active": np.nan, "is_silencer": np.nan,
        "sequence": d["seq"].astype(str).str.upper().values,
    })
    out = out[out["sequence"].str.len() > 0].dropna(subset=["log2_ratio"]).reset_index(drop=True)
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    path = Path(args.outdir) / f"promoter_seq_{args.assay}.csv"
    out.to_csv(path, index=False)
    print(f"{args.assay}: {len(out)} elements  (activity mean={out['log2_ratio'].mean():.2f} "
          f"sd={out['log2_ratio'].std():.2f}); split "
          + ", ".join(f"{k}={int(v)}" for k, v in out['chrom'].map(
              {'chr7': 'test', 'chr8': 'val', 'chr1': 'train'}).value_counts().items()))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
