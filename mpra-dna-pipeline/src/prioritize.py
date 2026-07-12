"""
Prioritize functional regulatory variants.

Joins the ensemble variant-effect predictions to the library annotations
(rsID, target gene, overlapping TF motif, GWAS flag, eQTL flag) and ranks
variants by the model's predicted *functional probability* (mean of the two
assays' functional-classifier heads), with the predicted allelic effect and
direction. Produces:
  - results/variant_priority.csv  — every variant, annotated + scored, ranked
  - console shortlist of the top GWAS/eQTL-annotated candidates
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ASSAYS = ["Primary", "Organoid"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="results/variant_predictions_ens.csv")
    ap.add_argument("--variants", default="data/variants_alleles.csv")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    P = pd.read_csv(args.preds)
    V = pd.read_csv(args.variants)

    ann = (V[["insert_name", "rsid", "pos", "target_gene", "motif", "gwas", "qtl"]]
           .drop_duplicates("insert_name").set_index("insert_name"))
    for c in ["gwas", "qtl"]:
        ann[c] = pd.to_numeric(ann[c], errors="coerce").fillna(0).astype(int)

    df = P.merge(ann, on="insert_name", how="left")

    fp = df[[f"pred_functional_prob_{a}" for a in ASSAYS]].astype(float)
    df["functional_prob"] = fp.mean(axis=1)            # mean of the two assay heads
    df["functional_prob_max"] = fp.max(axis=1)
    eff = df[[f"pred_effect_{a}" for a in ASSAYS]].astype(float)
    df["pred_effect_mean"] = eff.mean(axis=1)
    df["pred_abs_effect"] = eff.abs().mean(axis=1)
    df["direction"] = np.where(df["pred_effect_mean"] >= 0, "increase", "decrease")
    # measured significance (min adj_p across assays), for reference only
    ap_cols = df[[f"adj_p_{a}" for a in ASSAYS]].astype(float)
    df["min_adj_p"] = ap_cols.min(axis=1)

    df = df.sort_values("functional_prob", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)

    cols = ["rank", "insert_name", "rsid", "chrom", "pos", "ref", "alt", "direction",
            "functional_prob", "functional_prob_max", "pred_effect_mean", "pred_abs_effect",
            "target_gene", "motif", "gwas", "qtl", "min_adj_p", "split",
            "measured_logFC_Primary", "measured_logFC_Organoid"]
    cols = [c for c in cols if c in df.columns]
    out = df[cols]
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    out.to_csv(Path(args.outdir) / "variant_priority.csv", index=False)

    # shortlist: GWAS- or eQTL-annotated candidates, highest functional prob
    cand = df[(df["gwas"] == 1) | (df["qtl"] == 1)].copy()
    print(f"{len(df)} variants scored; {len(cand)} carry a GWAS/eQTL annotation")
    print(f"\nTOP {args.top} FUNCTIONAL CANDIDATES (GWAS/eQTL-annotated, ranked by functional prob)")
    show = cand.head(args.top)
    for _, r in show.iterrows():
        tag = ("GWAS" if r["gwas"] == 1 else "") + ("+eQTL" if r["qtl"] == 1 else "")
        gene = r["target_gene"] if isinstance(r["target_gene"], str) and r["target_gene"].strip() else "-"
        mot = r["motif"] if isinstance(r["motif"], str) and r["motif"].strip() else "-"
        print(f"  {r['rsid'] or r['insert_name']:>12}  {r['chrom']}:{int(r['pos'])} {r['ref']}>{r['alt']}  "
              f"funcP={r['functional_prob']:.2f}  eff={r['pred_effect_mean']:+.3f}({r['direction']})  "
              f"gene={gene:<10} motif={mot:<8} [{tag}] split={r['split']}")
    # how many high-confidence, held-out candidates
    hi = df[(df["functional_prob"] >= 0.6) & (df["split"] == "test")]
    print(f"\n{len(hi)} variants with functional_prob>=0.60 in the held-out test set")
    print(f"saved {args.outdir}/variant_priority.csv")


if __name__ == "__main__":
    main()
