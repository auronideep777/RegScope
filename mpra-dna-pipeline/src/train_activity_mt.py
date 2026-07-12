"""
Multi-task activity model: activity regression + is_active + is_SILENCER.

Adds a silencer head to the shared encoder so the model explicitly learns
repression grammar, not just activation. Silencer labels exist only for the
Primary assay (is_silencer, ~6% positive), so that head is Primary-supervised
and masked elsewhere — the existing masked-loss machinery handles it.

Isolated on purpose: it subclasses RegNetDNA and writes models/activity_mt.pt,
touching none of the existing scripts. Reports activity Spearman, is-active
AUROC, and the new is-silencer AUROC on held-out chromosomes.

    python src/train_activity_mt.py --activity data/activity_seq.csv \
        --epochs 60 --seeds 3 --channels 192 --blocks 6 --batch 512 --outdir results
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from common import batch_one_hot, chrom_split, spearman, pearson, r2, auroc

ASSAYS = ["Primary", "Organoid"]


def rc_np(x):
    """Reverse-complement a (B,C,L) batch. Channels 0-3 are one-hot (ACGT->TGCA);
    channels 4+ are antisymmetric non-B tracks (g4/imotif/rloop) that NEGATE on
    the opposite strand. Length axis reversed for all."""
    C = x.shape[1]
    if C == 4:
        return x[:, ::-1, ::-1].copy()
    idx = [3, 2, 1, 0] + list(range(4, C))
    xr = x[:, idx, ::-1].copy()
    xr[:, 4:, :] *= -1.0
    return xr


def build(df, seq_len, feats=()):
    seq = df.groupby("insert_name")["sequence"].first()
    chrom = df.groupby("insert_name")["chrom"].first()
    w = pd.DataFrame({"sequence": seq, "chrom": chrom})
    for a in ASSAYS:
        sub = df[df["assay"] == a].set_index("insert_name")
        w[f"y_{a}"] = sub["log2_ratio"]
        w[f"act_{a}"] = sub["is_active"] if "is_active" in sub else np.nan
        w[f"sil_{a}"] = sub["is_silencer"] if "is_silencer" in sub else np.nan
    w = w.reset_index()
    w = w[w["sequence"].astype(str).str.len() > 0].reset_index(drop=True)
    X = batch_one_hot(w["sequence"].tolist(), seq_len)
    if feats:
        from g4 import build_nonb_channels
        NB = build_nonb_channels(w["sequence"].tolist(), seq_len, feats)   # (N,K,L)
        X = np.concatenate([X, NB], axis=1).astype(np.float32)
    Y = np.stack([w[f"y_{a}"].values for a in ASSAYS], 1).astype(np.float32)
    A = np.stack([w[f"act_{a}"].values for a in ASSAYS], 1).astype(np.float32)
    S = np.stack([pd.to_numeric(w[f"sil_{a}"], errors="coerce").values for a in ASSAYS], 1).astype(np.float32)
    return w, X, Y, A, S


def make_model(cfg, in_ch=4):
    import torch.nn as nn
    from models_best import RegNetDNA
    import torch

    class RegNetDNAMT(RegNetDNA):
        def __init__(self, in_ch=4, stem_k=15, **kw):
            super().__init__(**kw)
            self.in_ch = in_ch
            if in_ch != 4:                       # rebuild stem conv for extra channels
                ch = self.stem[0].out_channels
                self.stem[0] = nn.Conv1d(in_ch, ch, stem_k, padding=stem_k // 2)
            self.sil = nn.Linear(self.reg.in_features, self.n_assays)

        def _rc(self, x):
            if self.in_ch == 4:
                return torch.flip(x, dims=(1, 2))
            idx = torch.tensor([3, 2, 1, 0] + list(range(4, self.in_ch)), device=x.device)
            xr = x[:, idx, :].flip(-1).clone()
            xr[:, 4:, :] = -xr[:, 4:, :]
            return xr

        def forward(self, x, rc_average=False):
            f = self.embed(x)
            if rc_average:
                f = 0.5 * (f + self.embed(self._rc(x)))
            h = self.head(f)
            return self.reg(h), self.cls(h), self.sil(h)
    return RegNetDNAMT(in_ch=in_ch, n_assays=len(ASSAYS), **cfg)


def predict3(model, X, device, rc=True, batch=512):
    import torch
    model.eval(); R = []; C = []; S = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i + batch]).float().to(device)
            reg, cls, sil = model(xb, rc_average=rc)
            R.append(reg.cpu().numpy()); C.append(cls.cpu().numpy()); S.append(sil.cpu().numpy())
    sig = lambda z: 1 / (1 + np.exp(-z))
    return np.concatenate(R), sig(np.concatenate(C)), sig(np.concatenate(S))


def train_one(X, Y, A, S, tr, va, mu, sd, cfg, args, seed, device, in_ch=4):
    import torch, torch.nn as nn
    torch.manual_seed(seed); np.random.seed(seed)
    Yz = (Y - mu) / sd
    ym = np.isfinite(Y)
    model = make_model(cfg, in_ch=in_ch).to(device)
    _ws = getattr(args, "warm_start", None)
    if _ws:
        import torch as _t
        _sd = _t.load(_ws, map_location=device, weights_only=False)["state_dicts"][0]
        _m = model.state_dict()
        model.load_state_dict({k: v for k, v in _sd.items() if k in _m and _m[k].shape == v.shape}, strict=False)
        print("    warm-started encoder from", _ws, flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    warm = args.warmup
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda e: (e + 1) / warm if e < warm else
        0.5 * (1 + np.cos(np.pi * (e - warm) / max(1, args.epochs - warm))))
    huber = nn.HuberLoss(reduction="none", delta=1.0)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    idx = np.where(tr)[0]
    best, best_state, bad = -1e9, None, 0
    for ep in range(args.epochs):
        model.train(); np.random.shuffle(idx)
        for i in range(0, len(idx), args.batch):
            b = idx[i:i + args.batch]
            xb = X[b].copy()
            fl = np.random.rand(len(b)) < 0.5
            if fl.any():
                xb[fl] = rc_np(xb[fl])
            xb = torch.from_numpy(xb).float().to(device)
            yb = torch.from_numpy(np.nan_to_num(Yz[b])).float().to(device)
            mb = torch.from_numpy(ym[b].astype(np.float32)).to(device)
            ab = torch.from_numpy(np.nan_to_num(A[b])).float().to(device)
            amb = torch.from_numpy(np.isfinite(A[b]).astype(np.float32)).to(device)
            sb = torch.from_numpy(np.nan_to_num(S[b])).float().to(device)
            smb = torch.from_numpy(np.isfinite(S[b]).astype(np.float32)).to(device)
            reg, cls, sil = model(xb)
            lr_ = (huber(reg, yb) * mb).sum() / mb.sum().clamp(min=1)
            lc = (bce(cls, ab) * amb).sum() / amb.sum().clamp(min=1)
            ls = (bce(sil, sb) * smb).sum() / smb.sum().clamp(min=1)
            loss = lr_ + 0.3 * lc + 0.3 * ls
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        sched.step()
        pv, _, _ = predict3(model, X[va], device)
        vs = float(np.nanmean([spearman(Y[va][:, j], pv[:, j]) for j in range(len(ASSAYS))]))
        if vs > best:
            best, best_state, bad = vs, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
        print(f"  seed{seed} ep{ep:02d} val_spearman={vs:.4f} (best {best:.4f})", flush=True)
        if bad >= args.patience:
            print("  early stop", flush=True); break
    model.load_state_dict(best_state)
    return model


def main():
    import torch
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
    ap.add_argument("--g4", action="store_true", help="append G4Hunter channel")
    ap.add_argument("--imotif", action="store_true", help="append i-motif (C-rich) channel")
    ap.add_argument("--rloop", action="store_true", help="append R-loop (GC-skew) channel")
    ap.add_argument("--tag", default="", help="output filename suffix")
    ap.add_argument("--assays", default="Primary,Organoid",
                    help="comma-separated assay/cell-type names (task heads)")
    ap.add_argument("--extra", nargs="*", default=[],
                    help="extra activity CSVs to concatenate (e.g. promoter_seq_WTC11.csv)")
    ap.add_argument("--warm-start", default=None,
                    help="pretrained .pt to transfer matching encoder weights from")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    global ASSAYS
    ASSAYS = [a.strip() for a in args.assays.split(",") if a.strip()]
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    print(f"assays (task heads): {ASSAYS}")
    cfg = {"channels": args.channels, "n_blocks": args.blocks, "dropout": args.dropout}

    feats = [f for f, on in [("g4", args.g4), ("imotif", args.imotif), ("rloop", args.rloop)] if on]
    frames = [pd.read_csv(args.activity)] + [pd.read_csv(e) for e in args.extra]
    df = pd.concat(frames, ignore_index=True)
    df = df[df["assay"].isin(ASSAYS)]
    w, X, Y, A, S = build(df, args.seq_len, feats=feats)
    in_ch = X.shape[1]
    print(f"{len(w)} inserts; input channels={in_ch} (non-B: {feats or 'none'}); "
          "silencer positives: " + ", ".join(f"{a}={int(np.nansum(S[:, j]))}" for j, a in enumerate(ASSAYS)))
    tr, va, te = chrom_split(w["chrom"])
    ym = np.isfinite(Y)
    mu = np.array([np.nanmean(Y[tr][:, j][ym[tr][:, j]]) for j in range(len(ASSAYS))], np.float32)
    sd = np.array([np.nanstd(Y[tr][:, j][ym[tr][:, j]]) + 1e-8 for j in range(len(ASSAYS))], np.float32)

    t0 = time.time()
    models = [train_one(X, Y, A, S, tr, va, mu, sd, cfg, args, s, device, in_ch=in_ch)
              for s in range(args.seeds)]
    print(f"trained {len(models)} model(s) in {(time.time()-t0)/60:.1f} min")

    reg = np.mean([predict3(m, X[te], device)[0] for m in models], 0) * sd + mu
    cls = np.mean([predict3(m, X[te], device)[1] for m in models], 0)
    sil = np.mean([predict3(m, X[te], device)[2] for m in models], 0)
    metrics = {}
    for j, a in enumerate(ASSAYS):
        metrics[a] = {"spearman": spearman(Y[te][:, j], reg[:, j]),
                      "r2": r2(Y[te][:, j], reg[:, j]),
                      "auroc_is_active": auroc(A[te][:, j], cls[:, j]),
                      "auroc_is_silencer": auroc(S[te][:, j], sil[:, j])}
    print("\nMULTI-TASK HELD-OUT METRICS (activity + is_active + is_silencer)"
          + (f" + non-B: {feats}" if feats else ""))
    print(json.dumps(metrics, indent=2))
    Path(args.outdir).mkdir(parents=True, exist_ok=True); Path("models").mkdir(exist_ok=True)
    tag = args.tag or ("_" + "".join(f[0] for f in feats) if feats else "")
    torch.save({"state_dicts": [m.state_dict() for m in models], "mu": mu, "sd": sd,
                "seq_len": args.seq_len, "assays": ASSAYS, "cfg": cfg,
                "multitask": True, "feats": feats, "in_ch": in_ch},
               f"models/activity_mt{tag}.pt")
    json.dump(metrics, open(Path(args.outdir) / f"metrics_mt{tag}.json", "w"), indent=2)
    print(f"saved models/activity_mt{tag}.pt + results/metrics_mt{tag}.json")


if __name__ == "__main__":
    main()
