#!/usr/bin/env python3
"""
run_demo.py — generate a realistic synthetic variant cohort, run the full
RegulatoryScope structure decomposition, and write results_demo.json.

The cohort plants real structural motifs (G4, i-motif, Z-DNA, R-loop, repeats) so
discrete events actually occur, and builds a measured_effect that depends on the
feature deltas + a GC confound + noise. Purpose: prove the pipeline end-to-end and
produce a schema-valid results.json the dashboard can load. NOT real biology.
"""
import numpy as np
from regscope_structure.features import score_variant, FAMILIES
from regscope_structure import decompose

RNG = np.random.default_rng(7)
BASES = "ACGT"; W = 41; C = W//2   # window length / centre index

def rand_seq(n, gc=0.42):
    p = [(1-gc)/2, gc/2, gc/2, (1-gc)/2]  # A,C,G,T
    return "".join(RNG.choice(list(BASES), size=n, p=p))

def implant(seq, motif, at):
    return seq[:at] + motif + seq[at+len(motif):]

def make_window(kind):
    s = rand_seq(W)
    if kind == "g4":
        s = implant(s, "GGGAGGGAGGGAGGG", C-7)
    elif kind == "imotif":
        s = implant(s, "CCCACCCTCCCACCC", C-7)
    elif kind == "zdna":
        s = implant(s, "CGCGCGCGCGCGCG", C-6)
    elif kind == "rloop":
        s = implant(s, "GGGGCGGGGCGGGGCGGGG", C-9)
    elif kind == "cruciform":
        arm = rand_seq(7)
        s = implant(s, arm + "AT" + _rc(arm), C-8)
    elif kind == "mirror":
        arm = "".join(RNG.choice(list("AG"), 7))   # homopurine
        s = implant(s, arm + "A" + arm[::-1], C-8)
    return s

_CMP = str.maketrans("ACGT","TGCA")
def _rc(s): return s.translate(_CMP)[::-1]

def mutate_center(seq):
    ref = seq[C]; alt = RNG.choice([b for b in BASES if b != ref])
    return seq, seq[:C] + alt + seq[C+1:]

KIND_MIX = (["g4"]*12 + ["imotif"]*6 + ["zdna"]*8 + ["rloop"]*8 +
            ["cruciform"]*6 + ["mirror"]*6 + ["bg"]*54)  # ~100 weights

def build_arm(n, weights, target_scale, gc_weight, seed):
    rng = np.random.default_rng(seed)
    kinds = rng.choice(KIND_MIX, size=n)
    records = []; refs = []
    for k in kinds:
        ref = make_window("bg" if k=="bg" else k)
        ref, alt = mutate_center(ref)
        records.append(score_variant(ref, alt)); refs.append(ref)
    # standardise per-family deltas, build effect
    fams = list(FAMILIES.keys())
    D = {f: np.array([r[f]["delta"] for r in records]) for f in fams}
    def z(v):
        s = np.std(v); return (v-np.mean(v))/s if s>0 else v*0
    gc = np.array([r["_base"]["gc"] for r in records])
    eff = np.zeros(n)
    for f in fams: eff += weights[f]*z(D[f])
    eff += gc_weight*(gc-np.mean(gc))/ (np.std(gc) or 1)
    eff += rng.normal(0, 1.0, n)                      # noise dominates (realistic)
    eff = eff/np.std(eff)*target_scale
    return records, eff

def main():
    # MPRA arm: composition matters (GC confound), shape/sidd/nuc lead.
    # Weights are small vs unit noise -> realistic small ΔR2 increments.
    w_mpra = dict(PGS=0.10, PIS=0.08, SHP=0.20, SID=0.14, ZDA=0.05,
                  RLP=0.06, CTX=0.04, NUC=0.13)
    rec_m, eff_m = build_arm(1200, w_mpra, target_scale=0.11, gc_weight=0.55, seed=11)

    # caQTL arm: weak composition signal, nucleosome + shape lead
    w_caqtl = dict(PGS=0.05, PIS=0.04, SHP=0.10, SID=0.07, ZDA=0.04,
                   RLP=0.06, CTX=0.03, NUC=0.16)
    rec_c, eff_c = build_arm(1500, w_caqtl, target_scale=0.045, gc_weight=0.06, seed=22)

    bundle = decompose.run({
        "mpra_transcription": {"records": rec_m, "effect": eff_m},
        "caqtl_accessibility": {"records": rec_c, "effect": eff_c},
    }, out_path="web/results_demo.json", n_boot=400)

    for arm, d in bundle["arms"].items():
        m = d["measured"]
        print(f"\n== {arm} ==  n={m['n_variants']}  base R2={m['nested_r2']['base']:.4f}"
              f"  +struct ΔR2={m['nested_r2']['incremental_structure']:.5f}")
        print("  structure-altering:", m["n_structure_altering"],
              " partial ρ=%.3f p=%s" % (m["partial_corr_structure"]["partial_spearman"],
                                        m["partial_corr_structure"]["p"]))
        for f, fb in d["families"].items():
            ev = fb["events"]; rg = fb["rigor"]
            print(f"    {f}: ΔR2={fb['dR2']:+.5f}  events={ev}  gc={rg['gc']}  dir={rg['dir']:.2f}")
    print("\nWrote web/results_demo.json")

if __name__ == "__main__":
    import os; os.makedirs("web", exist_ok=True); main()
