"""
Two-stage transfer: PRETRAIN the shared encoder on a large auxiliary MPRA
(e.g. WTC11), then FINE-TUNE on cortex (Primary+Organoid). The goal is to
inherit general regulatory grammar from the big set while ending up fully
specialised on cortex — the proper way to benefit from an off-target library
without the capacity dilution of equal-weight multi-task.

Stage 1 trains a model on the pretrain assay(s); its ENCODER (stem, conv tower,
attention pooling, shared head-MLP) is transferred into a fresh cortex model
(output heads are re-initialised since the assay set differs), which is then
fine-tuned on cortex only. Reports cortex held-out Spearman vs the 0.614/0.595
cortex-only baseline.

    python src/train_transfer.py \
        --pretrain-activity data/promoter_seq_WTC11.csv --pretrain-assays WTC11 \
        --activity data/activity_seq.csv --assays Primary,Organoid \
        --imotif --pretrain-epochs 40 --finetune-epochs 60 --seeds 3 --outdir results
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import train_activity_mt as tam
from common import chrom_split, spearman, r2, auroc


def prep(files, assays, seq_len, feats):
    tam.ASSAYS = list(assays)
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[df["assay"].isin(assays)]
    w, X, Y, A, S = tam.build(df, seq_len, feats=feats)
    tr, va, te = chrom_split(w["chrom"])
    ym = np.isfinite(Y)
    mu = np.array([np.nanmean(Y[tr][:, j][ym[tr][:, j]]) for j in range(len(assays))], np.float32)
    sd = np.array([np.nanstd(Y[tr][:, j][ym[tr][:, j]]) + 1e-8 for j in range(len(assays))], np.float32)
    return w, X, Y, A, S, tr, va, te, mu, sd


def train_model(X, Y, A, S, tr, va, mu, sd, cfg, device, in_ch, epochs, lr,
                seed, patience=12, init_state=None, freeze_stem=False):
    import torch, torch.nn as nn
    torch.manual_seed(seed); np.random.seed(seed)
    Yz = (Y - mu) / sd; ym = np.isfinite(Y)
    model = tam.make_model(cfg, in_ch=in_ch).to(device)
    if init_state is not None:
        msd = model.state_dict()
        filt = {k: v for k, v in init_state.items() if k in msd and msd[k].shape == v.shape}
        model.load_state_dict(filt, strict=False)
        print(f"    transferred {len(filt)}/{len(msd)} encoder tensors (output heads re-initialised)")
    if freeze_stem:
        for p in model.stem.parameters():
            p.requires_grad = False
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=2e-4)
    warm = 4
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda e: (e + 1) / warm if e < warm else 0.5 * (1 + np.cos(np.pi * (e - warm) / max(1, epochs - warm))))
    huber = nn.HuberLoss(reduction="none", delta=1.0); bce = nn.BCEWithLogitsLoss(reduction="none")
    idx = np.where(tr)[0]; best, best_state, bad = -1e9, None, 0
    for ep in range(epochs):
        model.train(); np.random.shuffle(idx)
        for i in range(0, len(idx), 512):
            b = idx[i:i + 512]; xb = X[b].copy()
            fl = np.random.rand(len(b)) < 0.5
            if fl.any(): xb[fl] = tam.rc_np(xb[fl])
            xb = torch.from_numpy(xb).float().to(device)
            yb = torch.from_numpy(np.nan_to_num(Yz[b])).float().to(device)
            mb = torch.from_numpy(ym[b].astype(np.float32)).to(device)
            ab = torch.from_numpy(np.nan_to_num(A[b])).float().to(device)
            amb = torch.from_numpy(np.isfinite(A[b]).astype(np.float32)).to(device)
            sb = torch.from_numpy(np.nan_to_num(S[b])).float().to(device)
            smb = torch.from_numpy(np.isfinite(S[b]).astype(np.float32)).to(device)
            reg, cls, sil = model(xb)
            loss = (huber(reg, yb) * mb).sum() / mb.sum().clamp(min=1) \
                + 0.3 * (bce(cls, ab) * amb).sum() / amb.sum().clamp(min=1) \
                + 0.3 * (bce(sil, sb) * smb).sum() / smb.sum().clamp(min=1)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        sched.step()
        pv, _, _ = tam.predict3(model, X[va], device)
        vs = float(np.nanmean([spearman(Y[va][:, j], pv[:, j]) for j in range(Y.shape[1])]))
        if vs > best:
            best, best_state, bad = vs, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
        print(f"    seed{seed} ep{ep:02d} val_spearman={vs:.4f} (best {best:.4f})", flush=True)
        if bad >= patience:
            print("    early stop", flush=True); break
    model.load_state_dict(best_state)
    return model


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrain-activity", required=True, nargs="+",
                    help="one or more converted MPRA CSVs to pretrain on (e.g. WTC11 K562 HepG2)")
    ap.add_argument("--pretrain-assays", default="WTC11",
                    help="comma-separated assay names present in the pretrain CSVs")
    ap.add_argument("--activity", default="data/activity_seq.csv")
    ap.add_argument("--assays", default="Primary,Organoid")
    ap.add_argument("--seq-len", type=int, default=270)
    ap.add_argument("--pretrain-epochs", type=int, default=40)
    ap.add_argument("--finetune-epochs", type=int, default=60)
    ap.add_argument("--finetune-lr", type=float, default=8e-4)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--channels", type=int, default=192)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--imotif", action="store_true")
    ap.add_argument("--g4", action="store_true")
    ap.add_argument("--freeze-stem", action="store_true")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    cfg = {"channels": args.channels, "n_blocks": args.blocks, "dropout": args.dropout}
    feats = [f for f, on in [("g4", args.g4), ("imotif", args.imotif)] if on]
    pre_assays = [a.strip() for a in args.pretrain_assays.split(",") if a.strip()]
    cortex_assays = [a.strip() for a in args.assays.split(",") if a.strip()]

    # ---- Stage 1: pretrain on auxiliary (WTC11) ----
    print(f"\n=== STAGE 1: pretrain encoder on {pre_assays} ===")
    _, Xp, Yp, Ap, Sp, trp, vap, tep, mup, sdp = prep(args.pretrain_activity, pre_assays, args.seq_len, feats)
    in_ch = Xp.shape[1]
    print(f"pretrain: {Xp.shape[0]} elements, in_ch={in_ch}")
    pre_model = train_model(Xp, Yp, Ap, Sp, trp, vap, mup, sdp, cfg, device, in_ch,
                            args.pretrain_epochs, 2e-3, seed=0)
    enc_state = {k: v.cpu().clone() for k, v in pre_model.state_dict().items()}

    # ---- Stage 2: fine-tune on cortex (transfer encoder) ----
    print(f"\n=== STAGE 2: fine-tune on {cortex_assays} (encoder transferred) ===")
    _, X, Y, A, S, tr, va, te, mu, sd = prep([args.activity], cortex_assays, args.seq_len, feats)
    tam.ASSAYS = cortex_assays
    t0 = time.time()
    models = [train_model(X, Y, A, S, tr, va, mu, sd, cfg, device, in_ch,
                          args.finetune_epochs, args.finetune_lr, seed=s,
                          init_state=enc_state, freeze_stem=args.freeze_stem)
              for s in range(args.seeds)]
    print(f"fine-tuned {len(models)} model(s) in {(time.time()-t0)/60:.1f} min")

    reg = np.mean([tam.predict3(m, X[te], device)[0] for m in models], 0) * sd + mu
    cls = np.mean([tam.predict3(m, X[te], device)[1] for m in models], 0)
    sil = np.mean([tam.predict3(m, X[te], device)[2] for m in models], 0)
    metrics = {}
    for j, a in enumerate(cortex_assays):
        metrics[a] = {"spearman": spearman(Y[te][:, j], reg[:, j]), "r2": r2(Y[te][:, j], reg[:, j]),
                      "auroc_is_active": auroc(A[te][:, j], cls[:, j]),
                      "auroc_is_silencer": auroc(S[te][:, j], sil[:, j])}
    print("\nPRETRAIN->FINETUNE HELD-OUT CORTEX METRICS")
    print(json.dumps(metrics, indent=2))
    print("(compare Primary/Organoid Spearman to cortex-only baseline 0.614 / 0.595)")
    Path(args.outdir).mkdir(parents=True, exist_ok=True); Path("models").mkdir(exist_ok=True)
    torch.save({"state_dicts": [m.state_dict() for m in models], "mu": mu, "sd": sd,
                "seq_len": args.seq_len, "assays": cortex_assays, "cfg": cfg,
                "feats": feats, "in_ch": in_ch, "pretrained_on": pre_assays},
               "models/activity_transfer.pt")
    json.dump(metrics, open(Path(args.outdir) / "metrics_transfer.json", "w"), indent=2)
    print("saved models/activity_transfer.pt + results/metrics_transfer.json")


if __name__ == "__main__":
    main()
