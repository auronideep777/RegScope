"""
Export the model's learned motifs (and the JASPAR2024 CORE vertebrate database)
in MEME minimal format, so you can run the OFFICIAL MEME-Tomtom for rigorous
E-values:

    tomtom -no-ssc -oc tomtom_out results/learned_motifs_Primary.meme results/jaspar2024_core_vert.meme

or upload results/learned_motifs_Primary.meme to https://meme-suite.org/meme/tools/tomtom
(select the JASPAR2024 CORE vertebrates database there).

    python src/export_meme.py --model models/activity_best.pt \
        --activity data/activity_seq.csv --assay Primary --outdir results
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from common import batch_one_hot
from jaspar_match import extract_filter_pwms, consensus, ic_trim

HEADER = ("MEME version 4\n\nALPHABET= ACGT\n\nstrands: + -\n\n"
          "Background letter frequencies\nA 0.25 C 0.25 G 0.25 T 0.25\n\n")


def write_meme(path, motifs):
    """motifs: list of (name, pwm[4,w], nsites)"""
    with open(path, "w") as fh:
        fh.write(HEADER)
        for name, pwm, nsites in motifs:
            p = np.clip(pwm, 1e-9, 1); p = p / p.sum(0, keepdims=True)
            w = p.shape[1]
            fh.write(f"MOTIF {name}\n")
            fh.write(f"letter-probability matrix: alength= 4 w= {w} nsites= {int(nsites)} E= 0\n")
            for j in range(w):
                fh.write("  " + " ".join(f"{p[b, j]:.6f}" for b in range(4)) + "\n")
            fh.write("\n")


def main():
    import torch
    from models_best import RegNetDNA
    from pyjaspar import jaspardb
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/activity_best.pt")
    ap.add_argument("--activity", default="data/activity_seq.csv")
    ap.add_argument("--assay", default="Primary")
    ap.add_argument("--n-seq", type=int, default=3000)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    ck = torch.load(args.model, map_location="cpu", weights_only=False)
    cfg = ck.get("cfg", {"channels": 192, "n_blocks": 6, "dropout": 0.2})
    enc = RegNetDNA(n_assays=len(ck["assays"]), **cfg)
    enc.load_state_dict((ck.get("state_dicts") or [ck["state_dict"]])[0]); enc.eval()

    df = pd.read_csv(args.activity)
    df = df[df["assay"] == args.assay].sort_values("log2_ratio", ascending=False)
    X = batch_one_hot(df["sequence"].astype(str).head(args.n_seq).tolist(), ck["seq_len"])
    filts, _ = extract_filter_pwms(enc, X, torch)
    learned = [(f"filter{f['filter']}_{consensus(f['pwm'])}", ic_trim(f["pwm"]), f["n"]) for f in filts]
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    qpath = Path(args.outdir) / f"learned_motifs_{args.assay}.meme"
    write_meme(qpath, learned)
    print(f"wrote {qpath}  ({len(learned)} learned motifs)")

    jdb = jaspardb(release="JASPAR2024")
    jm = jdb.fetch_motifs(collection="CORE", tax_group=["vertebrates"])
    jaspar = []
    for m in jm:
        P = np.array([list(m.pwm[b]) for b in "ACGT"], float)
        jaspar.append((f"{m.matrix_id}_{m.name}", P, 20))
    jpath = Path(args.outdir) / "jaspar2024_core_vert.meme"
    write_meme(jpath, jaspar)
    print(f"wrote {jpath}  ({len(jaspar)} JASPAR motifs)")
    print("\nRun official Tomtom locally (after installing the MEME suite):")
    print(f"  tomtom -no-ssc -oc tomtom_{args.assay} {qpath} {jpath}")
    print("or upload the learned_motifs file to https://meme-suite.org/meme/tools/tomtom")


if __name__ == "__main__":
    main()
