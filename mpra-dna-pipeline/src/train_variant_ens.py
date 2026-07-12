"""
Variant-effect model v2 — ensemble + confidence-weighted training.

Two upgrades over train_variant_best.py:
  1. ENSEMBLE WARM-START: the activity model is a 3-seed ensemble. Here we
     warm-start and fine-tune a Siamese variant model from EACH seed, then
     average their predicted allelic effects (and functional-probabilities).
     Ensembling is the most reliable way to lift a noisy signal.
  2. CONFIDENCE-WEIGHTED LOSS: instead of up-weighting significant variants by a
     flat factor, each variant's effect loss is weighted by its measured
     confidence  w = 1 + clip(-log10(adj_p), 0, 6).  Inert variants (adj_p~1)
     get w~1; a variant at adj_p=1e-6 gets w~7 — so the fit is driven by the
     variants that actually did something, without throwing the rest away.

Everything else (held-out chromosomes, masked losses, ISM+correction head,
functional-classification head) is identical to train_variant_best.py, whose
helpers we reuse.

    python src/train_variant_ens.py --variants data/variants_alleles.csv \
        --warm-start models/activity_best.pt --epochs 50 --outdir results
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

from common import chrom_split, spearman
from train_variant_best import build_variant_table, evaluate, predict_pairs, ASSAYS


def train_seed(state_dict, cfg, Xr_t, Xa_t, tr, va,
               y_delta, y_ref, y_alt, y_sig, w_conf, logfc, adjp,
               args, device, seed):
    import torch, torch.nn as nn
    from models_best import RegNetDNA
    torch.manual_seed(seed); np.random.seed(seed)
    enc = RegNetDNA(n_assays=len(ASSAYS), **cfg).to(device)
    enc.load_state_dict(state_dict)
    fdim = cfg["channels"] * 3
    vhead = nn.Sequential(
        nn.Linear(fdim * 3, 256), nn.GELU(), nn.Dropout(0.2),
        nn.Linear(256, 2 * len(ASSAYS))).to(device)
    nn.init.zeros_(vhead[-1].weight); nn.init.zeros_(vhead[-1].bias)

    def embed_rc(x):
        return 0.5 * (enc.embed(x) + enc.embed(torch.flip(x, dims=(1, 2))))

    def forward(xr, xa):
        fr, fa = embed_rc(xr), embed_rc(xa)
        sr = enc.reg(enc.head(fr)); sa = enc.reg(enc.head(fa))
        ism = sa - sr
        out = vhead(torch.cat([fa - fr, fr, fa], -1))
        eff = ism + out[:, :len(ASSAYS)]
        return eff, out[:, len(ASSAYS):], sr, sa

    params = list(enc.parameters()) + list(vhead.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    huber = nn.HuberLoss(reduction="none", delta=1.0)
    pos = float(np.nansum(y_sig.cpu().numpy()))
    neg = float(np.isfinite(y_sig.cpu().numpy()).sum() - pos)
    bce = nn.BCEWithLogitsLoss(reduction="none",
                               pos_weight=torch.tensor([neg / max(pos, 1)] * len(ASSAYS)).to(device))

    def masked(el, mask):
        m = mask.float()
        return (el * m).sum() / m.sum().clamp(min=1)

    idx_tr = np.where(tr)[0]
    best, best_state, bad, patience = -1e9, None, 0, 10
    for ep in range(args.epochs):
        enc.train(); vhead.train(); np.random.shuffle(idx_tr)
        for i in range(0, len(idx_tr), args.batch):
            b = idx_tr[i:i + args.batch]
            xr = Xr_t[b].to(device); xa = Xa_t[b].to(device)
            eff, siglogit, sr, sa = forward(xr, xa)
            dm = torch.isfinite(y_delta[b]).float()
            ld = (huber(eff, torch.nan_to_num(y_delta[b])) * w_conf[b] * dm).sum() / dm.sum().clamp(min=1)
            la = masked(huber(sr, torch.nan_to_num(y_ref[b])), torch.isfinite(y_ref[b])) \
                 + masked(huber(sa, torch.nan_to_num(y_alt[b])), torch.isfinite(y_alt[b]))
            ls = masked(bce(siglogit, torch.nan_to_num(y_sig[b])), torch.isfinite(y_sig[b]))
            loss = ld + args.anchor * la + args.sig_weight * ls
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0); opt.step()
        sched.step()
        ev, sv = predict_pairs(forward, Xr_t, Xa_t, va, device)
        ev = ev[va]; obsv = logfc[va]; sigv = (adjp[va] < 0.05)
        vs = float(np.nanmean([spearman(obsv[:, j][sigv[:, j]], ev[:, j][sigv[:, j]])
                               for j in range(2) if sigv[:, j].sum() >= 10]))
        if not np.isfinite(vs):
            vs = float(np.nanmean([spearman(obsv[:, j], ev[:, j]) for j in range(2)]))
        if vs > best:
            best, bad = vs, 0
            best_state = ({k: v.cpu().clone() for k, v in enc.state_dict().items()},
                          {k: v.cpu().clone() for k, v in vhead.state_dict().items()})
        else:
            bad += 1
        print(f"  seed{seed} epoch {ep:02d} val_signed_spearman(sig)={vs:.4f} (best {best:.4f})", flush=True)
        if bad >= patience:
            print("  early stop", flush=True); break

    if best_state:
        enc.load_state_dict(best_state[0]); vhead.load_state_dict(best_state[1])
    eff_all, sig_all = predict_pairs(forward, Xr_t, Xa_t, np.ones(len(Xr_t), bool), device)
    return eff_all, sig_all, (best_state if best_state else None)


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="data/variants_alleles.csv")
    ap.add_argument("--warm-start", default="models/activity_best.pt")
    ap.add_argument("--seq-len", type=int, default=270)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--anchor", type=float, default=0.2)
    ap.add_argument("--sig-weight", type=float, default=1.0)
    ap.add_argument("--conf-cap", type=float, default=6.0)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if args.threads and args.threads > 0:
        torch.set_num_threads(args.threads)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    var = pd.read_csv(args.variants)
    base, Xr, Xa = build_variant_table(var, args.seq_len)
    n = len(base); print(f"{n} variant loci")

    logfc = np.stack([base[f"logFC_{a}"].values for a in ASSAYS], 1).astype(np.float32)
    adjp = np.stack([base[f"adjp_{a}"].values for a in ASSAYS], 1).astype(np.float32)
    refr = np.stack([base[f"refr_{a}"].values for a in ASSAYS], 1).astype(np.float32)
    altr = np.stack([base[f"altr_{a}"].values for a in ASSAYS], 1).astype(np.float32)
    sig = (adjp < 0.05).astype(np.float32); sig[~np.isfinite(adjp)] = np.nan
    tr, va, te = chrom_split(base["chrom"])
    print(f"split train={int(tr.sum())} val={int(va.sum())} test={int(te.sum())}")

    ck = torch.load(args.warm_start, map_location=device, weights_only=False)
    cfg = ck.get("cfg", {"channels": 192, "n_blocks": 6, "dropout": 0.2})
    state_dicts = ck.get("state_dicts") or [ck["state_dict"]]
    mu = np.asarray(ck["mu"], np.float32); sdv = np.asarray(ck["sd"], np.float32)
    print(f"warm-start ensemble: {len(state_dicts)} activity seeds")

    # confidence weights  w = 1 + clip(-log10(adj_p), 0, cap)
    with np.errstate(divide="ignore", invalid="ignore"):
        conf = np.clip(-np.log10(adjp), 0, args.conf_cap)
    conf[~np.isfinite(conf)] = 0.0
    w_conf = torch.from_numpy(1.0 + conf).float().to(device)

    y_delta = torch.from_numpy(logfc / sdv).float().to(device)
    y_ref = torch.from_numpy((refr - mu) / sdv).float().to(device)
    y_alt = torch.from_numpy((altr - mu) / sdv).float().to(device)
    y_sig = torch.from_numpy(sig).float().to(device)
    Xr_t = torch.from_numpy(Xr).float(); Xa_t = torch.from_numpy(Xa).float()

    effs, sigs, states = [], [], []
    for s, sd in enumerate(state_dicts):
        print(f"=== variant model warm-started from activity seed {s} ===", flush=True)
        eff_all, sig_all, st = train_seed(sd, cfg, Xr_t, Xa_t, tr, va,
                                          y_delta, y_ref, y_alt, y_sig, w_conf,
                                          logfc, adjp, args, device, s)
        effs.append(eff_all); sigs.append(sig_all); states.append(st)
        rep_s = evaluate(logfc[te], adjp[te], eff_all[te], sig_all[te])
        print(f"  seed{s} test sign_acc_significant: "
              + ", ".join(f"{a}={rep_s[a].get('sign_acc_significant', float('nan')):.3f}" for a in ASSAYS))

    ens_eff = np.mean(effs, 0)
    ens_sig = np.mean(sigs, 0)                     # average logits
    report = evaluate(logfc[te], adjp[te], ens_eff[te], ens_sig[te])
    print("\nENSEMBLE VARIANT-EFFECT MODEL — held-out test (chr7/chr17)")
    print(json.dumps(report, indent=2))

    Path(args.outdir).mkdir(parents=True, exist_ok=True); Path("models").mkdir(exist_ok=True)
    json.dump(report, open(Path(args.outdir) / "variant_metrics_ens.json", "w"), indent=2)
    torch.save({"states": states, "cfg": cfg, "mu": mu, "sd": sdv,
                "seq_len": args.seq_len, "assays": ASSAYS}, "models/variant_ens.pt")

    out = base[["insert_name", "chrom", "off", "ref", "alt"]].rename(columns={"off": "var_offset"})
    for j, a in enumerate(ASSAYS):
        out[f"pred_effect_{a}"] = ens_eff[:, j] * sdv[j]
        out[f"pred_functional_prob_{a}"] = 1 / (1 + np.exp(-ens_sig[:, j]))
        out[f"measured_logFC_{a}"] = logfc[:, j]
        out[f"adj_p_{a}"] = adjp[:, j]
    out["split"] = np.where(te, "test", np.where(va, "val", "train"))
    out.to_csv(Path(args.outdir) / "variant_predictions_ens.csv", index=False)
    print("saved models/variant_ens.pt, results/variant_metrics_ens.json, "
          "results/variant_predictions_ens.csv")


if __name__ == "__main__":
    main()
