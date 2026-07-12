"""
Motif enrichment — do the learned motifs actually mark active vs inactive
elements? For each top filter we take its per-sequence max activation and test
how well it separates is_active=1 from is_active=0 oligos (AUROC + Mann-Whitney).
Ties each motif to its JASPAR TF label (from jaspar_match_<assay>.csv) and its
activating/repressive sign. This is the rigorous, model-independent check that
the motifs are real regulatory signal, not artifacts.

    python src/motif_enrich.py --model models/activity_best.pt \
        --activity data/activity_seq.csv --assay Primary --outdir results
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from common import batch_one_hot


def mannwhitney_auc(pos, neg):
    """AUROC via the rank-sum (Mann-Whitney U) statistic + normal-approx p."""
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    n1, n2 = len(pos), len(neg)
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    allv = np.concatenate([pos, neg])
    ranks = pd.Series(allv).rank().values
    R1 = ranks[:n1].sum()
    U1 = R1 - n1 * (n1 + 1) / 2
    auc = U1 / (n1 * n2)
    mu = n1 * n2 / 2
    sd = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12) + 1e-9
    z = (U1 - mu) / sd
    # two-sided p via normal approx
    from math import erfc, sqrt
    p = erfc(abs(z) / sqrt(2))
    return float(auc), float(p)


def main():
    import torch
    from models_best import RegNetDNA
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/activity_best.pt")
    ap.add_argument("--activity", default="data/activity_seq.csv")
    ap.add_argument("--assay", default="Primary")
    ap.add_argument("--n-seq", type=int, default=12000)
    ap.add_argument("--jaspar", default=None, help="jaspar_match_<assay>.csv for TF labels")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    ck = torch.load(args.model, map_location="cpu", weights_only=False)
    cfg = ck.get("cfg", {"channels": 192, "n_blocks": 6, "dropout": 0.2})
    enc = RegNetDNA(n_assays=len(ck["assays"]), **cfg)
    enc.load_state_dict((ck.get("state_dicts") or [ck["state_dict"]])[0]); enc.eval()

    df = pd.read_csv(args.activity)
    df = df[(df["assay"] == args.assay) & df["is_active"].notna()]
    # balanced-ish sample of active + inactive
    act = df[df["is_active"] == 1]; ina = df[df["is_active"] == 0]
    k = min(args.n_seq // 2, len(act), len(ina))
    sub = pd.concat([act.sample(k, random_state=0), ina.sample(k, random_state=0)])
    y = sub["is_active"].values.astype(int)
    X = batch_one_hot(sub["sequence"].astype(str).tolist(), ck["seq_len"])
    print(f"{args.assay}: {k} active + {k} inactive oligos")

    with torch.no_grad():
        H = enc.stem[2](enc.stem[1](enc.stem[0](torch.from_numpy(X).float()))).numpy()
        pred = enc(torch.from_numpy(X).float(), rc_average=True)[0].numpy()[:, ck["assays"].index(args.assay)]
    C = H.shape[1]
    fmax = H.max(2)                                   # (N, C) per-filter max activation
    corr = np.array([np.corrcoef(fmax[:, f], pred)[0, 1] for f in range(C)])

    tf_label = {}
    if args.jaspar and Path(args.jaspar).exists():
        j = pd.read_csv(args.jaspar)
        tf_label = dict(zip(j["filter"], j["best_TF"]))

    rows = []
    for f in range(C):
        auc, p = mannwhitney_auc(fmax[y == 1, f], fmax[y == 0, f])
        rows.append({"filter": f, "corr_activity": round(float(corr[f]), 3),
                     "auroc_active_vs_inactive": round(auc, 3), "pvalue": p,
                     "sign": "activating" if corr[f] > 0 else "repressive",
                     "best_TF": tf_label.get(f, "")})
    R = pd.DataFrame(rows)
    R["abs_enrich"] = (R["auroc_active_vs_inactive"] - 0.5).abs()
    R = R.sort_values("abs_enrich", ascending=False)
    R.to_csv(Path(args.outdir) / f"motif_enrichment_{args.assay}.csv", index=False)

    print(f"\nMotif enrichment in ACTIVE vs INACTIVE oligos ({args.assay}) — top 15:")
    print(f"{'filter':>6} {'TF':>12} {'sign':>11} {'AUROC':>7} {'p':>10}")
    for _, r in R.head(15).iterrows():
        print(f"{int(r['filter']):>6} {str(r['best_TF']):>12} {r['sign']:>11} "
              f"{r['auroc_active_vs_inactive']:>7.3f} {r['pvalue']:>10.1e}")
    n_sig = int((R["pvalue"] < 1e-3).sum())
    print(f"\n{n_sig}/{C} filters separate active vs inactive at p<1e-3")
    print(f"saved results/motif_enrichment_{args.assay}.csv")


if __name__ == "__main__":
    main()
