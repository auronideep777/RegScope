"""
Synthetic enhancer design by in-silico directed evolution.

Start from random DNA and let the activity model drive it uphill: at each step,
score every possible single-base change and apply the one that most increases
predicted activity (greedy hill-climb), until convergence. Produces novel
sequences the model predicts as strong enhancers, tracks the activity
trajectory, benchmarks the designs against the distribution of REAL measured
MPRA activity, and checks which activating motifs emerged.

    python src/design.py --model models/activity_best.pt --assay Primary \
        --n 8 --steps 45 --activity data/activity_seq.csv --outdir results
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from common import BASES, one_hot


def make_scorer(model, ai, device, seq_len, rc=False):
    import torch
    def score(onehots):                       # (N,4,L) -> (N,)
        model.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(onehots), 512):
                xb = torch.from_numpy(onehots[i:i + 512]).float().to(device)
                reg, _ = model(xb, rc_average=rc)
                out.append(reg[:, ai].cpu().numpy())
        return np.concatenate(out)
    return score


def evolve(x, score, steps):
    L = x.shape[1]
    cur = float(score(x[None])[0])
    traj = [cur]
    for _ in range(steps):
        muts, coords = [], []
        for pos in range(L):
            c = int(np.argmax(x[:, pos]))
            for b in range(4):
                if b == c:
                    continue
                m = x.copy(); m[:, pos] = 0; m[b, pos] = 1
                muts.append(m); coords.append((pos, b))
        s = score(np.stack(muts))
        j = int(np.argmax(s))
        if s[j] <= cur + 1e-4:
            break
        pos, b = coords[j]
        x = x.copy(); x[:, pos] = 0; x[b, pos] = 1
        cur = float(s[j]); traj.append(cur)
    return x, traj


def onehot_to_seq(x):
    return "".join(BASES[i] for i in np.argmax(x, 0))


def main():
    import torch
    from ism import load_model
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/activity_best.pt")
    ap.add_argument("--assay", default="Primary")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--steps", type=int, default=45)
    ap.add_argument("--activity", default="data/activity_seq.csv")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    device = "cpu"
    np.random.seed(0)
    model, mu, sd, assays, seq_len = load_model(args.model, device)
    ai = assays.index(args.assay)
    score = make_scorer(model, ai, device, seq_len)

    # real activity distribution (de-standardised units are the raw log2_ratio)
    real = pd.read_csv(args.activity)
    real = pd.to_numeric(real[real["assay"] == args.assay]["log2_ratio"], errors="coerce").dropna().values
    real_hi = np.percentile(real, 99)

    def to_units(z):
        return z * sd[ai] + mu[ai]

    rows, trajs, seqs = [], [], []
    for k in range(args.n):
        rnd = "".join(np.random.choice(list(BASES), seq_len))
        x0 = one_hot(rnd, seq_len)
        xf, traj = evolve(x0, score, args.steps)
        a0, af = to_units(traj[0]), to_units(traj[-1])
        pct = float((real < af).mean() * 100)
        rows.append({"design": f"design_{k+1}", "start_activity": round(a0, 3),
                     "final_activity": round(af, 3), "gain": round(af - a0, 3),
                     "steps": len(traj) - 1, "pctile_vs_real": round(pct, 1)})
        trajs.append([to_units(z) for z in traj]); seqs.append(onehot_to_seq(xf))
        print(f"design_{k+1}: {a0:+.2f} -> {af:+.2f} ({len(traj)-1} steps), "
              f"beats {pct:.1f}% of real oligos")

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(Path(args.outdir) / f"designs_{args.assay}.csv", index=False)
    with open(Path(args.outdir) / f"designs_{args.assay}.fasta", "w") as fh:
        for k, s in enumerate(seqs):
            fh.write(f">design_{k+1}_pred{rows[k]['final_activity']}\n{s}\n")
    finals = np.array([r["final_activity"] for r in rows])
    print(f"\nreal 99th-pctile activity = {real_hi:.2f}; designed mean = {finals.mean():.2f}, "
          f"max = {finals.max():.2f}")
    _plot(trajs, real, real_hi, Path(args.outdir) / f"designs_{args.assay}.png", args.assay)
    print(f"saved results/designs_{args.assay}.csv/.fasta/.png")


def _plot(trajs, real, real_hi, path, assay):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [2, 1]})
    for t in trajs:
        a1.plot(t, alpha=0.8)
    a1.axhline(real_hi, ls="--", color="k", lw=1, label="real 99th pctile")
    a1.set_xlabel("evolution step"); a1.set_ylabel("predicted activity (log2 RNA/DNA)")
    a1.set_title(f"In-silico directed evolution — {assay}"); a1.legend(fontsize=8)
    a2.hist(real, bins=60, color="#bbb", label="real oligos")
    for t in trajs:
        a2.axvline(t[-1], color="crimson", lw=1)
    a2.set_xlabel("activity"); a2.set_title("designs (red) vs real"); a2.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


if __name__ == "__main__":
    main()
