"""
Motif interpretation — what regulatory grammar did the activity model learn?

The first conv layer (k=15, 192 filters) acts as a bank of learnable
position-weight matrices. For each filter we collect the input 15-mers at the
positions where it fires hardest (across many active oligos), stack them into a
PWM, and score it by information content. We also correlate each filter's
per-sequence activation with the model's predicted activity, so each motif is
tagged ACTIVATING (drives activity up) or REPRESSIVE (down). Top filters are
rendered as sequence logos.

    python src/motifs.py --model models/activity_best.pt \
        --activity data/activity_seq.csv --assay Primary --outdir results
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from common import batch_one_hot, BASES

# rough consensus of well-known TF motif families, for a first-pass label
KNOWN = {
    "SP1/KLF (GC-box)": "GGGGCGGGG", "KLF (GT-box)": "GGGGTGGGG",
    "AP-1 (FOS/JUN)": "TGACTCA", "E-box (bHLH)": "CACGTG",
    "GATA": "AGATAA", "ETS": "CGGAAGT", "CCAAT/NFY": "CCAATCA",
    "TATA": "TATAAAA", "Homeobox": "TAATTA", "FOX": "TGTTTAC",
    "NR (half-site)": "AGGTCA", "CREB/ATF": "TGACGTCA", "NRF1": "GCGCATGCGC",
    "RFX": "GTTGCCATGGCAAC", "MEF2": "CTATTTATAG",
}


def consensus(pwm):
    return "".join(BASES[i] if pwm[:, j].max() > 0.5 else "n"
                   for j, i in enumerate(pwm.argmax(0)))


def best_known(cons):
    c = cons.upper().replace("N", "")
    best, score = "-", 0
    def rc(s):
        return s.translate(str.maketrans("ACGT", "TGCA"))[::-1]
    for name, k in KNOWN.items():
        for probe in (c, rc(c)):
            for i in range(max(1, len(probe) - len(k) + 1)):
                w = probe[i:i + len(k)]
                if len(w) < 4:
                    continue
                m = sum(a == b for a, b in zip(w, k[:len(w)]))
                if m / max(len(w), len(k)) > score:
                    score, best = m / max(len(w), len(k)), name
    return best if score >= 0.75 else "-"


def pwm_to_ic_df(pwm):
    import pandas as pd
    p = np.clip(pwm, 1e-9, 1)
    p = p / p.sum(0, keepdims=True)
    ic = (2 + (p * np.log2(p)).sum(0))          # per-position information (bits)
    heights = (p * ic).T                         # (L, 4)
    return pd.DataFrame(heights, columns=list(BASES))


def main():
    import torch
    from models_best import RegNetDNA
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/activity_best.pt")
    ap.add_argument("--activity", default="data/activity_seq.csv")
    ap.add_argument("--assay", default="Primary")
    ap.add_argument("--n-seq", type=int, default=4000)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    device = "cpu"

    ck = torch.load(args.model, map_location=device, weights_only=False)
    cfg = ck.get("cfg", {"channels": 192, "n_blocks": 6, "dropout": 0.2})
    enc = RegNetDNA(n_assays=len(ck["assays"]), **cfg).to(device)
    (ck.get("state_dicts") or [ck["state_dict"]])
    enc.load_state_dict((ck.get("state_dicts") or [ck["state_dict"]])[0]); enc.eval()
    ai = ck["assays"].index(args.assay)

    df = pd.read_csv(args.activity)
    df = df[df["assay"] == args.assay]
    if "is_active" in df:
        df = df.sort_values("log2_ratio", ascending=False)
    seqs = df["sequence"].astype(str).head(args.n_seq).tolist()
    X = batch_one_hot(seqs, ck["seq_len"])
    Xt = torch.from_numpy(X).float()

    # first-layer conv activations aligned to input (conv+BN+GELU, pre-pool)
    conv, bn, act = enc.stem[0], enc.stem[1], enc.stem[2]
    with torch.no_grad():
        H = act(bn(conv(Xt))).numpy()            # (N, C, L)
        pred = enc(Xt, rc_average=True)[0][:, ai].numpy()   # predicted activity
    N, C, L = H.shape
    pad = conv.kernel_size[0] // 2

    # correlation of each filter's max activation with predicted activity
    filt_max = H.max(2)                            # (N, C)
    corr = np.array([np.corrcoef(filt_max[:, f], pred)[0, 1] for f in range(C)])

    rows = []
    for f in range(C):
        a = H[:, f, :]
        thr = 0.7 * a.max()
        if thr <= 0:
            continue
        flat = np.argwhere(a > thr)
        if len(flat) < 30:
            continue
        # cap to the 400 strongest to bound cost
        if len(flat) > 400:
            vals = a[flat[:, 0], flat[:, 1]]
            flat = flat[np.argsort(-vals)[:400]]
        wins = []
        for s, p in flat:
            lo, hi = p - pad, p + pad + 1
            if lo >= 0 and hi <= L:
                wins.append(X[s, :, lo:hi])
        if len(wins) < 30:
            continue
        pwm = np.stack(wins).mean(0)              # (4, k)
        p = np.clip(pwm, 1e-9, 1); p = p / p.sum(0, keepdims=True)
        ic = float((2 + (p * np.log2(p)).sum(0)).sum())
        rows.append({"filter": f, "ic": ic, "n": len(wins), "corr": corr[f],
                     "cons": consensus(pwm), "pwm": pwm})

    # rank by influence on predicted activity (the functionally relevant filters)
    rows.sort(key=lambda r: -abs(r["corr"]))
    top = rows[:args.top]
    print(f"extracted {len(rows)} informative filters; "
          f"top {len(top)} by |correlation with predicted activity|:\n")
    for r in top:
        tag = "ACTIVATING" if r["corr"] > 0 else "repressive"
        km = best_known(r["cons"])
        print(f"  filter {r['filter']:3d}  act->activity r={r['corr']:+.2f} [{tag:10s}]  "
              f"IC={r['ic']:4.1f}bits  n={r['n']:3d}  consensus={r['cons']}"
              + (f"  ~{km}" if km != '-' else ""))

    _plot_logos(top, Path(args.outdir) / f"motifs_{args.assay}.png", args.assay)
    # save a table too
    pd.DataFrame([{k: v for k, v in r.items() if k != "pwm"} for r in rows]).to_csv(
        Path(args.outdir) / f"motifs_{args.assay}.csv", index=False)
    print(f"\nsaved results/motifs_{args.assay}.png and results/motifs_{args.assay}.csv")


def _plot_logos(top, path, assay):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import logomaker
    n = len(top); ncol = 2; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 1.6 * nrow))
    axes = np.array(axes).reshape(-1)
    for ax, r in zip(axes, top):
        logomaker.Logo(pwm_to_ic_df(r["pwm"]), ax=ax, color_scheme="classic")
        tag = "activating" if r["corr"] > 0 else "repressive"
        km = best_known(r["cons"])
        lbl = f"filter {r['filter']} · r={r['corr']:+.2f} ({tag})" + (f" · ~{km}" if km != "-" else "")
        ax.set_title(lbl, fontsize=8)
        ax.set_ylim(0, 2); ax.set_xticks([]); ax.set_yticks([0, 1, 2])
        ax.set_ylabel("bits", fontsize=7)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(f"Learned first-layer motifs — activity model — {assay}", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


if __name__ == "__main__":
    main()
