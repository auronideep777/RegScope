# RegulatoryScope

**Read DNA → predict regulatory activity → ask what a single base does — then explain it through the full structural grammar of the genome.**

RegulatoryScope is a browser-based suite that pairs a trained sequence-to-activity model with an eleven-family structural decomposition, so a non-coding variant's predicted effect comes with the *why*: which structural axis (G-quadruplex, i-motif, DNA shape, duplex destabilization, Z-DNA, R-loop, cruciform, triplex/H-DNA, nucleosome) and which transcription-factor motif a single base perturbs — with statistics, citations, and exports, and nothing to install.

Built for the Built-with-Claude: Life Sciences (Gladstone) research track on the Deng et al. developing human cortex MPRA (*Science* 2024, eadh0559).

**Created by Ash0723.**

---

## Live demo

Enable GitHub Pages (Settings → Pages → Source: `main` / folder: `/docs`). The site entry point is `docs/index.html`.

- **Home** — the hub linking all four tools
- **MPRA Dashboard** — RegNetDNA results: DNA→activity, silencers, variant effects, saturation mutagenesis, TOMTOM motifs (held-out metrics)
- **RegScope** — base-resolution sequence scanner
- **SDM Studio** — site-directed-mutagenesis impact explorer (exportable table, per-mutation impact with references)
- **Additional Data** — six per-family structural dashboards + combined atlas

## Headline results (held-out chromosomes)

| Metric | Primary | Organoid |
|---|---|---|
| Activity Spearman (RegNetDNA) | 0.603 | 0.585 |
| Activity Spearman (best, +i-motif) | 0.614 | 0.595 |
| is-active AUROC / R² | 0.805 / 0.37 | 0.797 / 0.34 |
| is-silencer AUROC | ~0.80 | ~0.80 |
| Variant functional AUROC (ensemble) | 0.632 | 0.602 |

Data provenance verified byte-identical to the *Science* supplement (activity table = S1; variant library = S2). Structural engine independently validated: motif-detection AUROC 0.80–1.00 on controls; G4Hunter matches the literature (c-MYC +2.59, VEGF +2.79 vs random ~0.00); nearest-neighbour ΔG reproduces SantaLucia (1998) exactly. Full methods and a tested accuracy/reliability scorecard are in `docs/RegulatoryScope_Writeup.pdf`.

## Repository layout

```
docs/                         # the browser app (GitHub Pages site)
  index.html                  # entry point (Home)
  Home.dc.html                # Home (internal nav target)
  RegScope.html               # sequence scanner
  MPRA Dashboard.html         # RegNetDNA results dashboard
  SDM Studio.html             # site-directed-mutagenesis explorer
  Additional Data.html        # structural-family dashboards + atlas
  RegulatoryScope_Writeup.pdf # full technical write-up + references

pipeline/                     # reproducible structural-decomposition engine
  regscope_structure/
    features.py               # 11 feature scorers (+ tool-hook upgrade paths)
    decompose.py              # nested R², partial corr, GC-matched, bootstrap
  run_pipeline.py             # your variants.csv -> results.json
  run_demo.py                 # synthetic cohort -> results_demo.json
  generate_dashboards.py      # results.json -> family dashboards + index
  example_variants.csv        # sample input
  results_demo.json           # sample output

model/                        # RegNetDNA training/eval  (ADD YOUR CODE HERE)
data/                         # dataset provenance + how to obtain it
```

## Reproduce the structural pipeline

```bash
pip install -r requirements.txt
cd pipeline
python run_demo.py            # writes results_demo.json (synthetic demo cohort)
# or, on your own data:
python run_pipeline.py variants.csv -o results.json
python generate_dashboards.py
```

Every structural feature is a fast, transparent model with a documented path to its gold-standard tool (DNAshapeR, SIST/WebSIDD, Z-Hunt, QmRLFS-finder, non-B DB, NuPoP); see `docs/RegulatoryScope_Writeup.pdf` for the references.

## Dataset

Deng C, et al. (senior authors Ahituv N, Pollard KS, Nowakowski TJ). *Massively parallel characterization of regulatory elements in the developing human cortex.* Science. 2024;384(6698):eadh0559. https://doi.org/10.1126/science.adh0559 — see `data/README.md` for how to obtain the activity table (S1) and variant library (S2).


## License

MIT — see `LICENSE`.
