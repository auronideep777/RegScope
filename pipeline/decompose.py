"""Driver decomposition: does structure explain variant impact BEYOND GC and TF motifs?

Outputs results/decomposition.json + results/figures/*.png. Works on measured effects
(primary) or ISM effects (scale). This is the scientific crux; read the printed report.
"""
from __future__ import annotations
import json, os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, mannwhitneyu, binomtest


def _z(x):
    x = np.asarray(x, float); s = x.std()
    return (x - x.mean()) / s if s > 0 else x - x.mean()


def nested_r2(df, effect_col):
    """|effect| ~ baseline set -> +TF -> +structure. Return incremental R^2 at each step."""
    y = np.abs(df[effect_col].values)
    base = ["gc", "cpg", "pos_rel"] + (["baseline"] if df["baseline"].notna().any() else [])
    steps = {"base": base, "+TF": base + ["delta_pwm_max"],
             "+structure": base + ["delta_pwm_max", "delta_struct"]}
    r2 = {}
    for name, cols in steps.items():
        X = sm.add_constant(np.column_stack([_z(df[c]) for c in cols]))
        r2[name] = float(sm.OLS(y, X).fit().rsquared)
    return {"r2": r2,
            "incremental_TF": r2["+TF"] - r2["base"],
            "incremental_structure": r2["+structure"] - r2["+TF"]}


def partial_corr_structure(df, effect_col):
    """Spearman partial corr of delta_struct with |effect|, controlling GC, CpG, delta_pwm_max."""
    y = np.abs(df[effect_col].values)
    ctrl = np.column_stack([_z(df[c]) for c in ["gc", "cpg", "delta_pwm_max"]])
    ctrl = sm.add_constant(ctrl)
    ry = sm.OLS(y, ctrl).fit().resid
    rs = sm.OLS(_z(df["delta_struct"]), ctrl).fit().resid
    rho, p = spearmanr(rs, ry)
    return {"partial_spearman": float(rho), "p": float(p)}


def gc_matched_test(df, effect_col, gc_bins=5, act_bins=5):
    """Within GC x baseline bins, |effect| of structure-disrupting vs not (Mann-Whitney, pooled ranks)."""
    d = df.copy()
    d["gcbin"] = pd.qcut(d["gc"], gc_bins, duplicates="drop", labels=False)
    if d["baseline"].notna().any():
        d["abin"] = pd.qcut(d["baseline"].fillna(d["baseline"].median()),
                            act_bins, duplicates="drop", labels=False)
    else:
        d["abin"] = 0
    d["abseff"] = np.abs(d[effect_col])
    hit, non = [], []
    for _, g in d.groupby(["gcbin", "abin"]):
        h = g.loc[g["struct_disrupt"], "abseff"].values
        n = g.loc[~g["struct_disrupt"], "abseff"].values
        if len(h) and len(n):
            # center within bin to remove bin-level GC effect, then pool
            m = np.concatenate([h, n]).mean()
            hit += list(h - m); non += list(n - m)
    if not hit or not non:
        return {"note": "insufficient matched pairs"}
    u, p = mannwhitneyu(hit, non, alternative="greater")
    return {"n_struct": len(hit), "n_other": len(non),
            "median_struct_centered": float(np.median(hit)),
            "median_other_centered": float(np.median(non)),
            "mannwhitney_p_greater": float(p)}


def direction_test(df, effect_col):
    """Among structure-disrupting variants, does effect SIGN match the mechanistic prior?

    GC content predicts no sign; a consistent sign is a structure-specific signature.
    """
    d = df[df["struct_disrupt"] & (df["struct_dir_expect"] != 0)].copy()
    if len(d) < 5:
        return {"note": "too few structure-disrupting variants"}
    match = np.sign(d[effect_col].values) == np.sign(d["struct_dir_expect"].values)
    k = int(match.sum()); n = int(len(match))
    return {"n": n, "consistent": k, "frac": k / n,
            "binom_p": float(binomtest(k, n, 0.5, alternative="greater").pvalue)}


def bootstrap_incremental_structure(df, effect_col, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    idx = np.arange(len(df))
    for _ in range(n):
        s = df.iloc[rng.choice(idx, len(idx), replace=True)]
        try:
            vals.append(nested_r2(s, effect_col)["incremental_structure"])
        except Exception:
            continue
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"mean": float(np.mean(vals)), "ci95": [float(lo), float(hi)]}


def category_effects(df, effect_col):
    """Effect size by PGS/PIS identification event (disrupt/create/modulate/none).
    This is the identification-centric finding: do variants that DISRUPT or CREATE a
    PGS/PIS carry distinct, directional effects vs variants that touch no structure?"""
    out = {}
    base = np.abs(df.loc[df["struct_event"] == "none", effect_col])
    for ev, g in df.groupby("struct_event"):
        e = g[effect_col].values
        row = {"n": int(len(g)), "median_abs_effect": float(np.median(np.abs(e))),
               "mean_signed_effect": float(np.mean(e))}
        if ev != "none" and len(base) and len(g) >= 5:
            row["mannwhitney_p_vs_none"] = float(
                mannwhitneyu(np.abs(e), base, alternative="greater")[1])
        out[ev] = row
    return out


def run(features_path="results/variant_features.parquet", effect_col="measured_effect",
        config="config.yaml", out_json="results/decomposition.json",
        out_fig="results/figures/decomposition.png"):
    import yaml
    cfg = yaml.safe_load(open(config))["decompose"]
    df = pd.read_parquet(features_path).dropna(subset=[effect_col])
    out = {
        "effect_col": effect_col, "n_variants": int(len(df)),
        "n_structure_altering": int(df["struct_disrupt"].sum()),
        "event_counts": {k: int(v) for k, v in df["struct_event"].value_counts().items()},
        "category_effects": category_effects(df, effect_col),
        "nested_r2": nested_r2(df, effect_col),
        "partial_corr_structure": partial_corr_structure(df, effect_col),
        "gc_matched_test": gc_matched_test(df, effect_col, cfg["gc_bins"], cfg["activity_bins"]),
        "direction_test": direction_test(df, effect_col),
        "bootstrap_incremental_structure": bootstrap_incremental_structure(
            df, effect_col, cfg["n_bootstrap"]),
    }
    os.makedirs("results/figures", exist_ok=True)
    _figure(df, effect_col, out, out_fig)
    json.dump(out, open(out_json, "w"), indent=2)
    print(json.dumps(out, indent=2))
    return out


def _figure(df, effect_col, out, out_fig="results/figures/decomposition.png"):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    r2 = out["nested_r2"]["r2"]
    ax[0].bar(list(r2), list(r2.values()), color=["#8899aa", "#d98a3d", "#3b7a57"])
    ax[0].set_ylabel("R^2 of |effect|"); ax[0].set_title("Nested variance explained")
    dd = df[df["struct_disrupt"]]
    ax[1].scatter(dd["delta_struct"], np.abs(dd[effect_col]), s=8, alpha=0.5, color="#3b7a57")
    ax[1].set_xlabel("|structure delta|"); ax[1].set_ylabel("|effect|")
    ax[1].set_title("Structure-disrupting variants")
    fig.tight_layout(); fig.savefig(out_fig, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    run()
