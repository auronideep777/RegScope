#!/usr/bin/env python3
"""
run_pipeline.py — run the RegulatoryScope structure decomposition on YOUR variants.

Input: a CSV of variants with a reference window, the alternate allele, a measured
effect, and an arm label. Output: results.json the dashboard loads to turn every
family from illustrative into COMPUTED.

Required columns (flexible names accepted):
    arm            'mpra_transcription' | 'caqtl_accessibility' (or your own arm keys)
    ref_window     reference DNA window around the variant (e.g. 41-101 bp, variant centred)
    measured_effect signed effect of alt vs ref (log2FC for MPRA, accessibility Δ for caQTL)
  plus ONE of:
    alt_window     the full alternate-allele window, OR
    alt_allele (+ optional var_pos, else window centre)  to build the alt in place

Usage:
    python run_pipeline.py variants.csv -o web/results.json
    python run_pipeline.py variants.csv -o web/results.json --boot 1000

To upgrade any family to a gold-standard tool (DNAshapeR, SIST, Z-Hunt, QmRLFS-finder,
non-B DB, NuPoP), compute that feature externally and wire it into
regscope_structure/features.py (see load_dnashape_hook and the per-family scorers).
"""
import argparse, sys
import pandas as pd
from regscope_structure.features import score_variant
from regscope_structure import decompose

def build_alt(row):
    if "alt_window" in row and isinstance(row["alt_window"], str) and row["alt_window"]:
        return row["alt_window"]
    ref = row["ref_window"]
    pos = int(row["var_pos"]) if "var_pos" in row and pd.notna(row.get("var_pos")) else len(ref)//2
    alt = str(row["alt_allele"]).strip().upper()[0]
    return ref[:pos] + alt + ref[pos+1:]

def main():
    ap = argparse.ArgumentParser(description="RegulatoryScope structure decomposition")
    ap.add_argument("csv", help="variant table (see module docstring)")
    ap.add_argument("-o", "--out", default="results.json", help="output results.json")
    ap.add_argument("--boot", type=int, default=500, help="bootstrap resamples (default 500)")
    ap.add_argument("--tf-col", default=None, help="optional column with a TF-motif score")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    need = {"arm", "ref_window", "measured_effect"}
    missing = need - set(df.columns)
    if missing:
        sys.exit(f"ERROR: missing required columns: {sorted(missing)}")
    if "alt_window" not in df.columns and "alt_allele" not in df.columns:
        sys.exit("ERROR: provide either 'alt_window' or 'alt_allele' (+ optional 'var_pos').")

    df["ref_window"] = df["ref_window"].astype(str).str.upper()
    arm_inputs = {}
    for arm, sub in df.groupby("arm"):
        records, effect, tf = [], [], ([] if args.tf_col else None)
        for _, row in sub.iterrows():
            ref = row["ref_window"]; alt = build_alt(row)
            records.append(score_variant(ref, alt))
            effect.append(float(row["measured_effect"]))
            if args.tf_col is not None: tf.append(float(row[args.tf_col]))
        arm_inputs[arm] = {"records": records, "effect": effect, "tf": tf}
        print(f"[{arm}] scored {len(records)} variants")

    decompose.run(arm_inputs, out_path=args.out, n_boot=args.boot)
    print(f"Wrote {args.out} — load it into RegulatoryScope_Combined.html")

if __name__ == "__main__":
    main()
