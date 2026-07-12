"""
In-silico saturation mutagenesis (ISM) — the direct "what does a single base do"
readout, exactly the analysis Pollard/Ahituv-style MPRA models are used for.

For a given oligo it substitutes EVERY position with EACH of the other 3 bases,
scores each mutant with the RC-averaged activity model, and reports
   delta = activity(mutant) - activity(reference)
as a 4 x L mutation map (negative = the change lowers predicted activity). The
per-position "importance" is the max |delta| over the 3 possible substitutions —
peaks mark the bases the model thinks matter (motif footprints).

    python src/ism.py --model models/activity_best.pt --assay Primary \
        --variants data/variants_alleles.csv --pick-significant --outdir results
    python src/ism.py --model models/activity_best.pt --seq ACGT... --outdir results
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from common import one_hot, BASES


def load_model(path, device):
    import torch
    from models_best import RegNetDNA
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck.get("cfg", {"channels": 128, "n_blocks": 5, "dropout": 0.2})
    m = RegNetDNA(n_assays=len(ck["assays"]), **cfg).to(device)
    sd = ck.get("state_dicts") or [ck["state_dict"]]
    m.load_state_dict(sd[0]); m.eval()
    return m, ck["mu"], ck["sd"], ck["assays"], ck["seq_len"]


def score_batch(model, X, device, assay_idx, batch=256):
    import torch
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i + batch]).float().to(device)
            reg, _ = model(xb, rc_average=True)
            outs.append(reg[:, assay_idx].cpu().numpy())
    return np.concatenate(outs)


def ism_one(model, seq, seq_len, mu, sd, ai, device):
    seq = seq.upper()
    ref = one_hot(seq, seq_len)                       # (4, L)
    L = ref.shape[1]
    # reconstruct the (cropped/padded) reference string actually scored
    variants, coords = [ref], [(-1, -1)]
    for pos in range(L):
        col = ref[:, pos]
        if col.sum() == 0:            # padding / N
            continue
        cur = int(np.argmax(col))
        for b in range(4):
            if b == cur:
                continue
            mut = ref.copy(); mut[:, pos] = 0; mut[b, pos] = 1
            variants.append(mut); coords.append((pos, b))
    X = np.stack(variants).astype(np.float32)
    scores = score_batch(model, X, device, ai) * sd[ai] + mu[ai]   # log2 units
    ref_score = scores[0]
    delta = np.zeros((4, L), np.float32)
    for (pos, b), s in zip(coords[1:], scores[1:]):
        delta[b, pos] = s - ref_score
    importance = np.max(np.abs(delta), axis=0)
    return ref_score, delta, importance


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/activity_best.pt")
    ap.add_argument("--assay", default="Primary")
    ap.add_argument("--seq", default=None)
    ap.add_argument("--variants", default="data/variants_alleles.csv")
    ap.add_argument("--pick-significant", action="store_true",
                    help="pick the most significant held-out variant's oligo to map")
    ap.add_argument("--rsid", default=None, help="map a specific rsID's oligo")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    device = "cpu"
    model, mu, sd, assays, seq_len = load_model(args.model, device)
    ai = assays.index(args.assay)

    label, marks = "user_sequence", []
    if args.seq:
        seq = args.seq
    else:
        v = pd.read_csv(args.variants)
        v = v[v["assay"] == args.assay].copy()
        v["adj_p"] = pd.to_numeric(v["adj_p"], errors="coerce")
        if args.rsid:
            v = v[v["rsid"] == args.rsid]
            if not len(v):
                raise SystemExit(f"rsID {args.rsid} not found for assay {args.assay}")
        else:
            v = v[v["chrom"].isin(["chr7", "chr17"])]      # held-out
            v = v.sort_values("adj_p")
        row = v.iloc[0]
        seq = str(row["insert_sequence"])
        label = f'{row["insert_name"]}_{row["rsid"]}'
        marks = [(int(row["var_offset"]), row["ref"], row["alt"], float(row["logFC"]))]
        print(f"mapping {label}: {args.assay} adj_p={row['adj_p']:.2e} "
              f"measured logFC={row['logFC']:.3f}")

    ref_score, delta, importance = ism_one(model, seq, seq_len, mu, sd, ai, device)
    print(f"reference predicted activity ({args.assay}) = {ref_score:.3f} log2(RNA/DNA)")
    if marks:
        pos, rf, al, lfc = marks[0]
        # crop offset: one_hot centre-crops if seq longer than seq_len
        L0 = len(seq.upper())
        crop = (L0 - seq_len) // 2 if L0 > seq_len else 0
        p = pos - crop
        bidx = BASES.index(al) if al in BASES else None
        if 0 <= p < delta.shape[1] and bidx is not None:
            print(f"model's predicted effect for the ACTUAL variant "
                  f"({rf}>{al} @pos {pos}) = {delta[bidx, p]:+.3f}  "
                  f"(measured logFC {lfc:+.3f})")

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    import re as _re
    safe = _re.sub(r"[^A-Za-z0-9._-]", "_", f"{label}_{args.assay}")   # Windows-safe filename
    np.save(Path(args.outdir) / f"ism_delta_{safe}.npy", delta)
    _plot(delta, importance, seq_len, args.assay, label, marks, seq,
          Path(args.outdir) / f"ism_{safe}.png")
    print(f"saved results/ism_{safe}.png")


def _plot(delta, importance, seq_len, assay, label, marks, seq, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    L = delta.shape[1]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(min(22, L / 12 + 3), 5),
                                 gridspec_kw={"height_ratios": [1, 3]})
    a1.bar(range(L), importance, width=1.0, color="#444")
    a1.set_title(f"ISM mutation map — {assay} — {label}\n"
                 f"top: per-base importance (max |Δ|)   bottom: Δactivity per substitution")
    a1.set_xlim(-0.5, L - 0.5); a1.set_xticks([])
    im = a2.imshow(delta, aspect="auto", cmap="RdBu_r",
                   vmin=-np.abs(delta).max(), vmax=np.abs(delta).max(),
                   interpolation="nearest")
    a2.set_yticks(range(4)); a2.set_yticklabels(list(BASES))
    a2.set_xlabel("position in oligo"); a2.set_xlim(-0.5, L - 0.5)
    if marks:
        L0 = len(seq.upper()); crop = (L0 - seq_len) // 2 if L0 > seq_len else 0
        for pos, rf, al, lfc in marks:
            p = pos - crop
            if 0 <= p < L:
                a1.axvline(p, color="crimson", lw=1.2)
                a2.axvline(p, color="crimson", lw=1.2)
    fig.colorbar(im, ax=a2, fraction=0.02, pad=0.01, label="Δ log2(RNA/DNA)")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


if __name__ == "__main__":
    main()
