"""
Match the activity model's learned first-layer motifs to known TFs in JASPAR.

1. Extract per-filter PWMs from the first conv layer (activation-based, same as
   motifs.py) and tag each by its correlation with predicted activity.
2. Load JASPAR 2024 CORE vertebrate motifs (offline, via pyjaspar).
3. For each learned motif, do a Tomtom-style ungapped sliding alignment against
   every JASPAR motif (both orientations), scoring each overlap by the mean
   per-column Pearson correlation. Report the best-matching TF(s) with an
   empirical z-score against that query's full score distribution.

Not MEME-Tomtom's exact E-values, but a legitimate PWM-similarity search against
JASPAR. Outputs a table + a figure pairing each learned logo with its best TF.

    python src/jaspar_match.py --model models/activity_best.pt \
        --activity data/activity_seq.csv --assay Primary --top 12 --outdir results
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from common import batch_one_hot, BASES


# ----- learned-filter PWM extraction (activation based) --------------------- #
def extract_filter_pwms(model, X, torch, min_windows=30, cap=400):
    conv, bn, act = model.stem[0], model.stem[1], model.stem[2]
    with torch.no_grad():
        H = act(bn(conv(torch.from_numpy(X).float()))).numpy()      # (N,C,L)
        pred = model(torch.from_numpy(X).float(), rc_average=True)[0].numpy()
    N, C, L = H.shape
    pad = conv.kernel_size[0] // 2
    out = []
    for f in range(C):
        a = H[:, f, :]
        thr = 0.7 * a.max()
        if thr <= 0:
            continue
        flat = np.argwhere(a > thr)
        if len(flat) < min_windows:
            continue
        if len(flat) > cap:
            flat = flat[np.argsort(-a[flat[:, 0], flat[:, 1]])[:cap]]
        wins = [X[s, :, p - pad:p + pad + 1] for s, p in flat if 0 <= p - pad and p + pad + 1 <= L]
        if len(wins) < min_windows:
            continue
        pwm = np.stack(wins).mean(0)
        pwm = np.clip(pwm, 1e-9, 1); pwm = pwm / pwm.sum(0, keepdims=True)
        out.append({"filter": f, "pwm": pwm, "n": len(wins)})
    return out, pred


def consensus(pwm):
    return "".join(BASES[i] if pwm[:, j].max() > 0.5 else "n" for j, i in enumerate(pwm.argmax(0)))


def rc_pwm(pwm):
    return pwm[::-1, ::-1].copy()


def ic_trim(pwm, thresh=0.3):
    """Trim low-information flanks so the match focuses on the real motif core."""
    p = np.clip(pwm, 1e-9, 1)
    ic = 2 + (p * np.log2(p)).sum(0)
    keep = np.where(ic > thresh)[0]
    if len(keep) < 3:
        return pwm
    return pwm[:, keep.min():keep.max() + 1]


def col_corr(q, t):
    """mean per-column Pearson correlation over the best ungapped overlap."""
    Lq, Lt = q.shape[1], t.shape[1]
    best = -2.0
    for off in range(-(Lt - 3), Lq - 3 + 1):        # slide q over t, min overlap 3
        qs = max(0, -off); ts = max(0, off)
        ov = min(Lq - qs, Lt - ts)
        if ov < 4:
            continue
        cs = []
        for k in range(ov):
            a = q[:, qs + k]; b = t[:, ts + k]
            if a.std() < 1e-6 or b.std() < 1e-6:
                continue
            cs.append(np.corrcoef(a, b)[0, 1])
        if cs:
            best = max(best, float(np.mean(cs)) * (ov / max(Lq, Lt)) ** 0.5)  # favour longer overlaps
    return best


def main():
    import torch
    from models_best import RegNetDNA
    from pyjaspar import jaspardb
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/activity_best.pt")
    ap.add_argument("--activity", default="data/activity_seq.csv")
    ap.add_argument("--assay", default="Primary")
    ap.add_argument("--n-seq", type=int, default=4000)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    ck = torch.load(args.model, map_location="cpu", weights_only=False)
    cfg = ck.get("cfg", {"channels": 192, "n_blocks": 6, "dropout": 0.2})
    enc = RegNetDNA(n_assays=len(ck["assays"]), **cfg)
    enc.load_state_dict((ck.get("state_dicts") or [ck["state_dict"]])[0]); enc.eval()
    ai = ck["assays"].index(args.assay)

    df = pd.read_csv(args.activity)
    df = df[df["assay"] == args.assay].sort_values("log2_ratio", ascending=False)
    X = batch_one_hot(df["sequence"].astype(str).head(args.n_seq).tolist(), ck["seq_len"])

    print("extracting learned filter PWMs...")
    filts, pred = extract_filter_pwms(enc, X, torch)
    fm = np.stack([np.clip(f["pwm"], 0, 1).max(0).max() for f in filts])  # dummy
    # activity correlation per filter (recompute activations cheaply already have pred)
    with torch.no_grad():
        H = enc.stem[2](enc.stem[1](enc.stem[0](torch.from_numpy(X).float()))).numpy()
    for f in filts:
        f["corr"] = float(np.corrcoef(H[:, f["filter"], :].max(1), pred[:, ai])[0, 1])
        f["cons"] = consensus(f["pwm"])
    filts.sort(key=lambda d: -abs(d["corr"]))
    filts = filts[:args.top]

    print(f"loading JASPAR2024 CORE vertebrates...")
    jdb = jaspardb(release="JASPAR2024")
    jmotifs = jdb.fetch_motifs(collection="CORE", tax_group=["vertebrates"])
    jpwms = []
    for m in jmotifs:
        P = np.array([list(m.pwm[b]) for b in "ACGT"], dtype=float)
        P = np.clip(P, 1e-9, 1); P = P / P.sum(0, keepdims=True)
        jpwms.append((m.matrix_id, m.name, ic_trim(P)))
    print(f"  {len(jpwms)} JASPAR motifs")

    rows = []
    for f in filts:
        q = ic_trim(f["pwm"])
        scores = []
        for mid, name, P in jpwms:
            s = max(col_corr(q, P), col_corr(rc_pwm(q), P))
            scores.append(s)
        scores = np.array(scores)
        order = np.argsort(-scores)
        mu, sd = scores.mean(), scores.std() + 1e-9
        top3 = [(jpwms[i][1], jpwms[i][0], scores[i], (scores[i] - mu) / sd) for i in order[:3]]
        f["match"] = top3
        best = top3[0]
        tag = "activating" if f["corr"] > 0 else "repressive"
        rows.append({"filter": f["filter"], "consensus": f["cons"], "corr_activity": round(f["corr"], 3),
                     "tag": tag, "best_TF": best[0], "jaspar_id": best[1],
                     "match_score": round(best[2], 3), "z": round(best[3], 2),
                     "TF_2": top3[1][0], "TF_3": top3[2][0]})
        print(f"  filter {f['filter']:3d} [{tag:10s}] {f['cons']:15s} -> {best[0]:10s} "
              f"({best[1]}) score={best[2]:.3f} z={best[3]:+.1f}  | {top3[1][0]}, {top3[2][0]}")

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(Path(args.outdir) / f"jaspar_match_{args.assay}.csv", index=False)
    _plot(filts, jpwms, Path(args.outdir) / f"jaspar_match_{args.assay}.png", args.assay)
    print(f"\nsaved results/jaspar_match_{args.assay}.csv and .png")


def _to_logo_df(pwm):
    p = np.clip(pwm, 1e-9, 1); p = p / p.sum(0, keepdims=True)
    ic = 2 + (p * np.log2(p)).sum(0)
    return pd.DataFrame((p * ic).T, columns=list(BASES))


def _plot(filts, jpwms, path, assay):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import logomaker
    jmap = {mid: (name, P) for mid, name, P in jpwms}
    n = len(filts)
    fig, axes = plt.subplots(n, 2, figsize=(11, 1.3 * n))
    for i, f in enumerate(filts):
        tag = "activating" if f["corr"] > 0 else "repressive"
        logomaker.Logo(_to_logo_df(ic_trim(f["pwm"])), ax=axes[i, 0], color_scheme="classic")
        axes[i, 0].set_title(f"learned filter {f['filter']} · r={f['corr']:+.2f} ({tag})", fontsize=8)
        axes[i, 0].set_xticks([]); axes[i, 0].set_yticks([0, 2]); axes[i, 0].set_ylim(0, 2)
        tfn, tfid, sc, zz = f["match"][0]
        _, P = jmap[tfid]
        logomaker.Logo(_to_logo_df(P), ax=axes[i, 1], color_scheme="classic")
        axes[i, 1].set_title(f"JASPAR best: {tfn} ({tfid}) · score={sc:.2f} z={zz:+.1f}", fontsize=8)
        axes[i, 1].set_xticks([]); axes[i, 1].set_yticks([0, 2]); axes[i, 1].set_ylim(0, 2)
    fig.suptitle(f"Learned motifs matched to JASPAR2024 — {assay}", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


if __name__ == "__main__":
    main()
