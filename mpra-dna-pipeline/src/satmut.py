"""
Saturation-mutagenesis maps + quantitative summary for the top prioritized
candidates. For each element it scores every position x every base with the
activity model and reports, per element:
  - reference predicted activity
  - the single most-disruptive substitution (position, base, delta)
  - the ACTUAL variant's delta and its percentile rank among all possible
    single-base changes (is the real variant a high-impact site?)
  - whether the top-impact positions fall inside the annotated TF motif window

    python src/satmut.py --model models/activity_best.pt \
        --candidates results/top_candidates.csv --variants data/variants_alleles.csv \
        --assay Primary --topn 8 --outdir results
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from common import BASES
from ism import load_model, ism_one, _plot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/activity_best.pt")
    ap.add_argument("--candidates", default="results/top_candidates.csv")
    ap.add_argument("--variants", default="data/variants_alleles.csv")
    ap.add_argument("--assay", default="Primary")
    ap.add_argument("--topn", type=int, default=8)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    device = "cpu"
    model, mu, sd, assays, seq_len = load_model(args.model, device)
    ai = assays.index(args.assay)

    V = pd.read_csv(args.variants)
    V = V[V["assay"] == args.assay]
    cand = pd.read_csv(args.candidates)
    rsids = list(dict.fromkeys(cand["rsid"].dropna().tolist()))[:args.topn]

    rows, panels = [], []
    for rs in rsids:
        sub = V[V["rsid"] == rs]
        if not len(sub):
            continue
        r = sub.iloc[0]
        seq = str(r["insert_sequence"]); off = int(r["var_offset"])
        ref, alt = r["ref"], r["alt"]
        ref_score, delta, importance = ism_one(model, seq, seq_len, mu, sd, ai, device)
        L = delta.shape[1]
        # actual variant effect
        vdelta = float(delta[BASES.index(alt), off]) if (0 <= off < L and alt in BASES) else np.nan
        # most disruptive substitution overall
        flat = delta.copy(); pos_min = np.unravel_index(np.argmin(flat), flat.shape)
        most_dis = float(flat[pos_min]); mb, mp = BASES[pos_min[0]], int(pos_min[1])
        # percentile of |variant| among all possible |single-base changes|
        allabs = np.abs(delta[delta != 0])
        pct = float((allabs < abs(vdelta)).mean() * 100) if np.isfinite(vdelta) and len(allabs) else np.nan
        # is the variant position itself a high-impact site (top 10% importance)?
        thr = np.nanpercentile(importance[importance > 0], 90) if (importance > 0).any() else np.inf
        var_is_hotspot = bool(importance[off] >= thr) if 0 <= off < L else False
        motif = r.get("motif", ""); motif = motif if isinstance(motif, str) else ""
        rows.append({"rsid": rs, "chrom": r["chrom"], "pos": int(r["pos"]),
                     "ref": ref, "alt": alt, "ref_activity": round(ref_score, 3),
                     "variant_delta": round(vdelta, 4) if np.isfinite(vdelta) else np.nan,
                     "variant_pctile_of_all_muts": round(pct, 1) if np.isfinite(pct) else np.nan,
                     "variant_is_hotspot": var_is_hotspot,
                     "most_disruptive_delta": round(most_dis, 4),
                     "most_disruptive_pos": mp, "most_disruptive_base": mb,
                     "motif": motif})
        panels.append((rs, delta, importance, off, ref, alt, ref_score, motif))
        # individual map
        marks = [(off, ref, alt, np.nan)]
        safe = f"satmut_{rs}_{args.assay}"
        np.save(Path(args.outdir) / f"{safe}.npy", delta)
        _plot(delta, importance, seq_len, args.assay, f"{rs} ({motif})" if motif else rs,
              marks, seq, Path(args.outdir) / f"{safe}.png")
        print(f"{rs:>12} {motif:>7} ref={ref_score:+.2f} var_delta={vdelta:+.3f} "
              f"(pctile {pct:.0f}%) hotspot={var_is_hotspot} "
              f"most_disruptive={mb}@{mp} ({most_dis:+.3f})")

    df = pd.DataFrame(rows)
    df.to_csv(Path(args.outdir) / f"satmut_summary_{args.assay}.csv", index=False)
    _grid(panels, seq_len, args.assay, Path(args.outdir) / f"satmut_grid_{args.assay}.png")
    print(f"\nsaved results/satmut_summary_{args.assay}.csv + satmut_grid_{args.assay}.png + per-variant maps")


def _grid(panels, seq_len, assay, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(11, 1.5 * n))
    axes = np.atleast_1d(axes)
    for ax, (rs, delta, imp, off, ref, alt, rscore, motif) in zip(axes, panels):
        vmax = np.abs(delta).max()
        ax.imshow(delta, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax.set_yticks(range(4)); ax.set_yticklabels(list(BASES), fontsize=6)
        ax.axvline(off, color="k", lw=0.8)
        ax.set_title(f"{rs}  {motif}  ref={rscore:+.2f}  variant {ref}>{alt}@{off}", fontsize=8)
        ax.set_xticks([])
    axes[-1].set_xlabel("position in oligo")
    fig.suptitle(f"Saturation mutagenesis — top candidates — {assay}\n(blue = substitution lowers predicted activity)", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


if __name__ == "__main__":
    main()
