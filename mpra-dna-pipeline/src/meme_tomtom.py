"""
Official-algorithm TOMTOM: match the model's learned motifs to JASPAR2024 using
the memelite pure-Python implementation of the MEME TOMTOM 'complete score'
algorithm (proper p-values, length/IC-aware, reverse-complement aware).

    python src/meme_tomtom.py --query results/learned_motifs_Primary.meme \
        --target results/jaspar2024_core_vert.meme --assay Primary --outdir results
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from memelite import tomtom
from memelite.io import read_meme


def load(path):
    m = read_meme(path)
    names = list(m.keys())
    pwms = [np.asarray(m[k], dtype=float) for k in names]
    return names, pwms


def bh_qvalues(p):
    p = np.asarray(p, float); n = len(p)
    order = np.argsort(p); ranked = p[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n); out[order] = np.clip(q, 0, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default="results/learned_motifs_Primary.meme")
    ap.add_argument("--target", default="results/jaspar2024_core_vert.meme")
    ap.add_argument("--assay", default="Primary")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    qn, Qs = load(args.query)
    tn, Ts = load(args.target)
    print(f"{len(Qs)} learned query motifs vs {len(Ts)} JASPAR targets — running TOMTOM...")
    p, scores, offsets, overlaps, strands = tomtom(Qs, Ts, reverse_complement=True)
    p = np.asarray(p)                                   # (n_query, n_target)

    rows = []
    for i, q in enumerate(qn):
        pv = p[i]
        q_bh = bh_qvalues(pv)                            # per-query BH across targets
        order = np.argsort(pv)
        best = order[0]
        top3 = ", ".join(tn[j].split("_")[-1] for j in order[:3])
        rows.append({"query": q, "best_TF": tn[best], "p_value": float(pv[best]),
                     "q_value": float(q_bh[best]), "top3_TFs": top3})
    df = pd.DataFrame(rows).sort_values("q_value")
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(args.outdir) / f"tomtom_{args.assay}.csv", index=False)

    sig = df[df["q_value"] < 0.05]
    print(f"\n{len(sig)}/{len(df)} learned motifs match a JASPAR TF at q<0.05")
    print("\nTop TOMTOM matches (by q-value):")
    print(f"{'query':<28}{'best TF':<16}{'p':>10}{'q':>10}   top3")
    for _, r in df.head(18).iterrows():
        tf = r["best_TF"].split("_")[-1]
        print(f"{r['query'][:27]:<28}{tf[:15]:<16}{r['p_value']:>10.1e}{r['q_value']:>10.1e}   {r['top3_TFs'][:40]}")
    print(f"\nsaved results/tomtom_{args.assay}.csv")


if __name__ == "__main__":
    main()
