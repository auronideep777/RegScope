"""
Best variant-effect model — "what does a single base do".

Why this is built the way it is (and why the naive approach gives ~0.03):
This library is 31k GWAS/eQTL variants; only ~1,389 (4.4%) reach adj.P<0.05 and
the median |logFC| is ~0.05. So ~95% of the labels are measurement noise around
zero. A model that predicts logFC for ALL variants is mostly predicting noise
and will always score near zero — that is a property of the DATA, not the model.
The scientifically honest, useful targets are: (a) get the DIRECTION right on
the variants that actually did something (significant / large-effect), and
(b) SEPARATE functional variants from inert ones (AUROC). We optimise and report
exactly those.

Model: a Siamese network. The activity encoder (warm-started from the trained
activity model, so it already knows regulatory grammar) scores the reference and
alternate oligo, RC-averaged. Predicted allelic effect =
   ISM delta (alt-ref through the activity head)  +  a learned variant-head
   correction  (initialised at 0, so we start exactly at the zero-shot ISM model
   and only improve it).
A second head classifies whether a variant is functional (significant).

Every loss term masks non-finite targets, so a stray NaN can never poison the
gradient (the failure mode that produced val=nan before).

    python src/train_variant_best.py --variants data/variants_alleles.csv \
        --warm-start models/activity_best.pt --epochs 40 --outdir results
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from common import batch_one_hot, chrom_split, spearman, pearson, sign_accuracy, auroc

ASSAYS = ["Primary", "Organoid"]
ACGT = set("ACGT")


def build_variant_table(var, seq_len):
    """One row per variant locus, with per-assay logFC / adj_p / allele ratios."""
    v = var.copy()
    v = v[v["ref"].isin(ACGT) & v["alt"].isin(ACGT)]
    v = v[v["insert_sequence"].astype(str).str.len() > 0]
    v["var_offset"] = pd.to_numeric(v["var_offset"], errors="coerce")
    v = v[v["var_offset"].notna()]; v["var_offset"] = v["var_offset"].astype(int)
    # keep rows whose insert base equals ref (already aligned upstream)
    ok = [0 <= o < len(s) and s[o] == r
          for s, o, r in zip(v["insert_sequence"], v["var_offset"], v["ref"])]
    v = v[ok]
    # collapse to one row per locus, carry per-assay measurements
    base = v.groupby("insert_name").agg(
        chrom=("chrom", "first"), seq=("insert_sequence", "first"),
        off=("var_offset", "first"), ref=("ref", "first"), alt=("alt", "first")).reset_index()
    for a in ASSAYS:
        sub = v[v["assay"] == a].set_index("insert_name")
        base[f"logFC_{a}"] = base["insert_name"].map(pd.to_numeric(sub["logFC"], errors="coerce"))
        base[f"adjp_{a}"] = base["insert_name"].map(pd.to_numeric(sub["adj_p"], errors="coerce"))
        base[f"refr_{a}"] = base["insert_name"].map(pd.to_numeric(sub.get("ref_ratio"), errors="coerce")) if "ref_ratio" in sub else np.nan
        base[f"altr_{a}"] = base["insert_name"].map(pd.to_numeric(sub.get("alt_ratio"), errors="coerce")) if "alt_ratio" in sub else np.nan
    # build ref / alt one-hot
    ref_seqs, alt_seqs = [], []
    for _, r in base.iterrows():
        s, o = str(r["seq"]), int(r["off"])
        ref_seqs.append(s)
        alt_seqs.append(s[:o] + r["alt"] + s[o + 1:])
    Xr = batch_one_hot(ref_seqs, seq_len)
    Xa = batch_one_hot(alt_seqs, seq_len)
    return base, Xr, Xa


def main():
    import torch, torch.nn as nn
    from models_best import RegNetDNA
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="data/variants_alleles.csv")
    ap.add_argument("--warm-start", default="models/activity_best.pt")
    ap.add_argument("--seq-len", type=int, default=270)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--anchor", type=float, default=0.2)
    ap.add_argument("--sig-weight", type=float, default=1.0)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if args.threads and args.threads > 0:
        torch.set_num_threads(args.threads)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    torch.manual_seed(0); np.random.seed(0)

    var = pd.read_csv(args.variants)
    base, Xr, Xa = build_variant_table(var, args.seq_len)
    n = len(base)
    print(f"{n} variant loci with resolved alleles")

    # targets
    logfc = np.stack([base[f"logFC_{a}"].values for a in ASSAYS], 1).astype(np.float32)
    adjp = np.stack([base[f"adjp_{a}"].values for a in ASSAYS], 1).astype(np.float32)
    refr = np.stack([base[f"refr_{a}"].values for a in ASSAYS], 1).astype(np.float32)
    altr = np.stack([base[f"altr_{a}"].values for a in ASSAYS], 1).astype(np.float32)
    sig = (adjp < 0.05).astype(np.float32); sig[~np.isfinite(adjp)] = np.nan
    print("finite logFC per assay:", [int(np.isfinite(logfc[:, j]).sum()) for j in range(2)],
          " significant:", [int(np.nansum(sig[:, j])) for j in range(2)])

    tr, va, te = chrom_split(base["chrom"])
    print(f"split train={int(tr.sum())} val={int(va.sum())} test={int(te.sum())}")

    # ---- warm-started encoder + normalisation ----
    ck = torch.load(args.warm_start, map_location=device, weights_only=False)
    cfg = ck.get("cfg", {"channels": 128, "n_blocks": 5, "dropout": 0.2})
    enc = RegNetDNA(n_assays=len(ASSAYS), **cfg).to(device)
    sd_list = ck.get("state_dicts") or [ck["state_dict"]]
    enc.load_state_dict(sd_list[0]); print(f"warm-started encoder from {args.warm_start}")
    mu = np.asarray(ck["mu"], np.float32); sdv = np.asarray(ck["sd"], np.float32)

    feat = args.__dict__  # noqa
    fdim = cfg["channels"] * 3
    vhead = nn.Sequential(
        nn.Linear(fdim * 3, 256), nn.GELU(), nn.Dropout(0.2),
        nn.Linear(256, 2 * len(ASSAYS)))          # [effect_correction(2), sig_logit(2)]
    nn.init.zeros_(vhead[-1].weight); nn.init.zeros_(vhead[-1].bias)  # start at ISM
    vhead = vhead.to(device)

    def embed_rc(x):
        return 0.5 * (enc.embed(x) + enc.embed(torch.flip(x, dims=(1, 2))))

    def forward(xr, xa):
        fr, fa = embed_rc(xr), embed_rc(xa)
        sr = enc.reg(enc.head(fr)); sa = enc.reg(enc.head(fa))   # activity in z-units
        ism = sa - sr
        out = vhead(torch.cat([fa - fr, fr, fa], -1))
        eff = ism + out[:, :len(ASSAYS)]                        # corrected effect (z-units)
        siglogit = out[:, len(ASSAYS):]
        return eff, siglogit, sr, sa

    # targets in z-units (per assay)
    sdv_t = torch.from_numpy(sdv).to(device)
    y_delta = torch.from_numpy(logfc / sdv).float().to(device)
    y_ref = torch.from_numpy((refr - mu) / sdv).float().to(device)
    y_alt = torch.from_numpy((altr - mu) / sdv).float().to(device)
    y_sig = torch.from_numpy(sig).float().to(device)
    Xr_t = torch.from_numpy(Xr).float(); Xa_t = torch.from_numpy(Xa).float()

    params = list(enc.parameters()) + list(vhead.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    huber = nn.HuberLoss(reduction="none", delta=1.0)
    # class imbalance for significance head
    pos = float(np.nansum(sig)); neg = float(np.isfinite(sig).sum() - pos)
    pw = torch.tensor([neg / max(pos, 1)] * len(ASSAYS)).to(device)
    bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pw)

    idx_tr = np.where(tr)[0]
    best, best_state, bad, patience = -1e9, None, 0, 10

    def masked(loss_el, mask):
        m = mask.float()
        return (loss_el * m).sum() / m.sum().clamp(min=1)

    for ep in range(args.epochs):
        enc.train(); vhead.train(); np.random.shuffle(idx_tr)
        for i in range(0, len(idx_tr), args.batch):
            b = idx_tr[i:i + args.batch]
            xr = Xr_t[b].to(device); xa = Xa_t[b].to(device)
            eff, siglogit, sr, sa = forward(xr, xa)
            dm = torch.isfinite(y_delta[b])
            # emphasise the ~5% informative (significant) variants in the delta loss
            w = torch.ones_like(y_delta[b])
            sg = torch.nan_to_num(y_sig[b])
            w = w + 4.0 * sg
            ld = ((huber(eff, torch.nan_to_num(y_delta[b])) * w) * dm.float()).sum() / dm.float().sum().clamp(min=1)
            # anchor absolute allele activity where measured (keeps encoder honest)
            la = masked(huber(sr, torch.nan_to_num(y_ref[b])), torch.isfinite(y_ref[b])) \
                 + masked(huber(sa, torch.nan_to_num(y_alt[b])), torch.isfinite(y_alt[b]))
            sm = torch.isfinite(y_sig[b])
            ls = masked(bce(siglogit, torch.nan_to_num(y_sig[b])), sm)
            loss = ld + args.anchor * la + args.sig_weight * ls
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0); opt.step()
        sched.step()
        # validation: signed spearman on significant variants (the meaningful signal)
        ev, sv = predict_pairs(forward, Xr_t, Xa_t, va, device)
        ev, sv = ev[va], sv[va]
        obsv = logfc[va]
        sigv = (adjp[va] < 0.05)
        vs = float(np.nanmean([spearman(obsv[:, j][sigv[:, j]], ev[:, j][sigv[:, j]])
                               for j in range(2) if sigv[:, j].sum() >= 10]))
        if not np.isfinite(vs):  # fall back to all-variant signed spearman early on
            vs = float(np.nanmean([spearman(obsv[:, j], ev[:, j]) for j in range(2)]))
        if vs > best:
            best = vs
            best_state = ({k: v.cpu().clone() for k, v in enc.state_dict().items()},
                          {k: v.cpu().clone() for k, v in vhead.state_dict().items()})
            bad = 0
        else:
            bad += 1
        print(f"epoch {ep:02d} val_signed_spearman(sig)={vs:.4f} (best {best:.4f})", flush=True)
        if bad >= patience:
            print("early stop", flush=True); break

    if best_state:
        enc.load_state_dict(best_state[0]); vhead.load_state_dict(best_state[1])

    # ---- held-out test report ----
    eff_te, sig_te = predict_pairs(forward, Xr_t, Xa_t, te, device)
    report = evaluate(logfc[te], adjp[te], eff_te[te], sig_te[te])
    print("\nVARIANT-EFFECT MODEL — held-out test (chr7/chr17)")
    print(json.dumps(report, indent=2))

    Path(args.outdir).mkdir(parents=True, exist_ok=True); Path("models").mkdir(exist_ok=True)
    torch.save({"enc": enc.state_dict(), "vhead": vhead.state_dict(), "cfg": cfg,
                "mu": mu, "sd": sdv, "seq_len": args.seq_len, "assays": ASSAYS},
               "models/variant_best.pt")
    json.dump(report, open(Path(args.outdir) / "variant_metrics_best.json", "w"), indent=2)

    # ---- full ranked table: predicted effect + functional score for every variant ----
    eff_all, sig_all = predict_pairs(forward, Xr_t, Xa_t, np.ones(n, bool), device)
    out = base[["insert_name", "chrom", "off", "ref", "alt"]].copy()
    out = out.rename(columns={"off": "var_offset"})
    for j, a in enumerate(ASSAYS):
        out[f"pred_effect_{a}"] = eff_all[:, j] * sdv[j]        # back to logFC units
        out[f"pred_functional_prob_{a}"] = 1 / (1 + np.exp(-sig_all[:, j]))
        out[f"measured_logFC_{a}"] = logfc[:, j]
        out[f"adj_p_{a}"] = adjp[:, j]
    out["split"] = np.where(te, "test", np.where(va, "val", "train"))
    out.to_csv(Path(args.outdir) / "variant_predictions_best.csv", index=False)
    print("saved models/variant_best.pt, results/variant_metrics_best.json, "
          "results/variant_predictions_best.csv")


def predict_pairs(forward, Xr_t, Xa_t, mask, device, batch=256):
    import torch
    idx = np.where(mask)[0] if mask.dtype == bool else mask
    effs, sigs = np.full((len(Xr_t), 2), np.nan, np.float32), np.full((len(Xr_t), 2), np.nan, np.float32)
    with torch.no_grad():
        for i in range(0, len(idx), batch):
            b = idx[i:i + batch]
            eff, sl, _, _ = forward(Xr_t[b].to(device), Xa_t[b].to(device))
            effs[b] = eff.cpu().numpy(); sigs[b] = sl.cpu().numpy()
    return effs, sigs


def evaluate(logfc, adjp, eff, sig):
    rep = {}
    for j, a in enumerate(ASSAYS):
        obs, pred, sl = logfc[:, j], eff[:, j], sig[:, j]
        ap = adjp[:, j]
        finite = np.isfinite(obs)
        d = {"n_test": int(finite.sum()),
             "spearman_all": spearman(obs, pred),
             "sign_acc_all": sign_accuracy(obs, pred)}
        issig = np.isfinite(ap) & (ap < 0.05)
        if issig.sum() >= 10:
            d["n_significant"] = int(issig.sum())
            d["spearman_significant"] = spearman(obs[issig], pred[issig])
            d["sign_acc_significant"] = sign_accuracy(obs[issig], pred[issig])
        thr = np.nanpercentile(np.abs(obs[finite]), 90) if finite.sum() else np.nan
        big = finite & (np.abs(obs) >= thr)
        if big.sum() >= 10:
            d["n_large_effect(top10%)"] = int(big.sum())
            d["sign_acc_large_effect"] = sign_accuracy(obs[big], pred[big])
            d["spearman_large_effect"] = spearman(obs[big], pred[big])
        # functional-variant separation: does the sig head / |effect| find the significant ones?
        lab = issig.astype(float); lab[~np.isfinite(ap)] = np.nan
        d["auroc_functional_sighead"] = auroc(lab, sl)
        d["auroc_functional_by_abs_effect"] = auroc(lab, np.abs(pred))
        rep[a] = d
    return rep


if __name__ == "__main__":
    main()
