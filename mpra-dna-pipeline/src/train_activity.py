"""
Train the best activity model (RegNetDNA) on insert-level MPRA activity.

Multi-task (Primary + Organoid), chromosome-held-out split, RC augmentation in
training + RC averaging at inference, cosine LR with warmup, masked losses so
missing-assay NaNs never poison the gradient. Optionally trains an ENSEMBLE of
seeds and averages them (test-time) for the strongest numbers.

    python src/train_activity.py --activity data/activity_seq.csv \
        --epochs 45 --seeds 2 --outdir results
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from common import batch_one_hot, chrom_split, spearman, pearson, r2, auroc

ASSAYS = ["Primary", "Organoid"]


def pivot_assays(df):
    seq = df.groupby("insert_name")["sequence"].first()
    chrom = df.groupby("insert_name")["chrom"].first()
    wide = pd.DataFrame({"sequence": seq, "chrom": chrom})
    for a in ASSAYS:
        sub = df[df["assay"] == a].set_index("insert_name")
        wide[f"y_{a}"] = sub["log2_ratio"]
        wide[f"act_{a}"] = sub["is_active"] if "is_active" in sub else np.nan
    return wide.reset_index()


def make_xy(wide, seq_len):
    X = batch_one_hot(wide["sequence"].tolist(), seq_len)
    Y = np.stack([wide[f"y_{a}"].values for a in ASSAYS], 1).astype(np.float32)
    A = np.stack([wide[f"act_{a}"].values for a in ASSAYS], 1).astype(np.float32)
    return X, Y, A


def train_one(X, Y, A, tr, va, mu, sd, args, seed, device):
    import torch, torch.nn as nn
    from models_best import RegNetDNA, predict
    torch.manual_seed(seed); np.random.seed(seed)
    Yz = (Y - mu) / sd
    ymask = np.isfinite(Y)
    model = RegNetDNA(n_assays=len(ASSAYS), channels=args.channels,
                      n_blocks=args.blocks, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    warm = args.warmup
    def lr_at(ep):
        if ep < warm:
            return (ep + 1) / warm
        p = (ep - warm) / max(1, args.epochs - warm)
        return 0.5 * (1 + np.cos(np.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    huber = nn.HuberLoss(reduction="none", delta=1.0)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    idx_tr = np.where(tr)[0]
    best, best_state, bad, patience = -1e9, None, 0, args.patience
    for ep in range(args.epochs):
        model.train(); np.random.shuffle(idx_tr)
        for i in range(0, len(idx_tr), args.batch):
            bi = idx_tr[i:i + args.batch]
            xb = X[bi].copy()
            flip = np.random.rand(len(bi)) < 0.5          # RC augmentation
            xb[flip] = xb[flip][:, ::-1, ::-1]
            xb = torch.from_numpy(xb).float().to(device)
            yb = torch.from_numpy(np.nan_to_num(Yz[bi])).float().to(device)
            mb = torch.from_numpy(ymask[bi].astype(np.float32)).to(device)
            ab = torch.from_numpy(np.nan_to_num(A[bi])).float().to(device)
            amb = torch.from_numpy(np.isfinite(A[bi]).astype(np.float32)).to(device)
            reg, cls = model(xb)
            lreg = (huber(reg, yb) * mb).sum() / mb.sum().clamp(min=1)
            lcls = (bce(cls, ab) * amb).sum() / amb.sum().clamp(min=1)
            loss = lreg + 0.3 * lcls
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()
        pv = predict(model, X[va], device, rc_average=True)
        vs = float(np.nanmean([spearman(Y[va][:, j], pv[:, j]) for j in range(len(ASSAYS))]))
        if vs > best:
            best, best_state, bad = vs, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
        print(f"  seed{seed} epoch {ep:02d} val_spearman={vs:.4f} (best {best:.4f}) lr={opt.param_groups[0]['lr']:.2e}", flush=True)
        if bad >= patience:
            print("  early stop", flush=True); break
    model.load_state_dict(best_state)
    return model


def main():
    import torch
    from models_best import predict
    ap = argparse.ArgumentParser()
    ap.add_argument("--activity", default="data/activity_seq.csv")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--seq-len", type=int, default=270)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--channels", type=int, default=192)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--tag", default="", help="suffix for output files (avoid clobbering)")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if args.threads and args.threads > 0:
        torch.set_num_threads(args.threads)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    df = pd.read_csv(args.activity)
    wide = pivot_assays(df)
    wide = wide[wide["sequence"].astype(str).str.len() > 0].reset_index(drop=True)
    print(f"{len(wide)} inserts; coverage " +
          ", ".join(f"{a}={int(wide[f'y_{a}'].notna().sum())}" for a in ASSAYS))
    tr, va, te = chrom_split(wide["chrom"])
    print(f"split train={int(tr.sum())} val={int(va.sum())} test={int(te.sum())}")
    X, Y, A = make_xy(wide, args.seq_len)
    ymask = np.isfinite(Y)
    mu = np.array([np.nanmean(Y[tr][:, j][ymask[tr][:, j]]) for j in range(len(ASSAYS))], np.float32)
    sd = np.array([np.nanstd(Y[tr][:, j][ymask[tr][:, j]]) + 1e-8 for j in range(len(ASSAYS))], np.float32)

    t0 = time.time()
    models = []
    for s in range(args.seeds):
        print(f"=== training seed {s} ===", flush=True)
        models.append(train_one(X, Y, A, tr, va, mu, sd, args, s, device))
    print(f"trained {len(models)} model(s) in {(time.time()-t0)/60:.1f} min")

    # ensemble test-time (RC-averaged) predictions
    def ens_reg(Xin):
        return np.mean([predict(m, Xin, device, rc_average=True, which="reg") for m in models], 0)
    def ens_cls(Xin):
        p = np.mean([1/(1+np.exp(-predict(m, Xin, device, rc_average=True, which="cls"))) for m in models], 0)
        return p
    pt = ens_reg(X[te]) * sd + mu
    ct = ens_cls(X[te])
    metrics = {}
    for j, a in enumerate(ASSAYS):
        yj, pj = Y[te][:, j], pt[:, j]
        metrics[a] = {"spearman": spearman(yj, pj), "pearson": pearson(yj, pj),
                      "r2": r2(yj, pj), "auroc_is_active": auroc(A[te][:, j], ct[:, j]),
                      "n_test": int(np.isfinite(yj).sum())}
    print("\nHELD-OUT TEST METRICS (ensemble, RC-averaged)")
    print(json.dumps(metrics, indent=2))

    Path(args.outdir).mkdir(parents=True, exist_ok=True); Path("models").mkdir(exist_ok=True)
    # save first model's weights (+ ensemble list) with normalisation stats
    import torch as T
    tag = args.tag
    T.save({"state_dicts": [m.state_dict() for m in models], "mu": mu, "sd": sd,
            "seq_len": args.seq_len, "assays": ASSAYS,
            "cfg": {"channels": args.channels, "n_blocks": args.blocks, "dropout": args.dropout}},
           f"models/activity_best{tag}.pt")
    json.dump(metrics, open(Path(args.outdir) / f"metrics_best{tag}.json", "w"), indent=2)
    _plot(Y[te], pt, Path(args.outdir) / f"pred_vs_obs_best{tag}.png")
    print(f"saved models/activity_best{tag}.pt + results/metrics_best{tag}.json")


def _plot(Y, P, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, len(ASSAYS), figsize=(5 * len(ASSAYS), 4.5))
    ax = np.atleast_1d(ax)
    for j, a in enumerate(ASSAYS):
        m = np.isfinite(Y[:, j]) & np.isfinite(P[:, j])
        ax[j].scatter(Y[m, j], P[m, j], s=4, alpha=0.3)
        ax[j].set_title(f"{a} rho={spearman(Y[:,j],P[:,j]):.3f}")
        ax[j].set_xlabel("measured log2(RNA/DNA)"); ax[j].set_ylabel("predicted")
    fig.tight_layout(); fig.savefig(path, dpi=130)


if __name__ == "__main__":
    main()
