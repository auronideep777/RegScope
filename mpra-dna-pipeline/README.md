# MPRA DNA Model — Reading DNA to Predict Regulatory Activity and Single-Base Effects

A sequence-to-function deep-learning pipeline on developing human **cortex MPRA** data
(Song, Pollard & Ahituv *et al.*, *Science*, `adh0559`). The model reads a raw DNA sequence,
predicts its regulatory activity, and answers *what a single base change does* — then extends
to variant-effect prediction, motif interpretation, synthetic design, and cross-cell transfer.

See **`report/final_report.pdf`** for the full write-up and **`RESULTS_SUMMARY.xlsx`** for all
result tables in one workbook.

## Headline results (held-out chromosomes: test = chr7+chr17, val = chr8+chr9)

| Task | Result |
|---|---|
| Activity — Spearman (Primary / Organoid) | **0.614 / 0.595** (≈2× k-mer baseline; prev CNN 0.558/0.528) |
| is-active / is-silencer AUROC | **0.81 / 0.80** |
| Variant effect — functional AUROC (P / O) | **0.632 / 0.602** |
| Motifs vs JASPAR (official TOMTOM, q<0.05) | **33 / 192** filters (CTCF, GATA1::TAL1, MAF, Pou5f1::Sox2, …) |
| Synthetic enhancer design | designs reach **top ~1%** predicted activity |
| Cross-cell (WTC11) transfer | **negative** — cortex-only is best (documented honestly) |
| Data provenance | **byte-identical** to Supplement S1 (activity) + S2 (variants) |

## Model

`RegNetDNA` (`src/models_best.py`): wide motif-detecting stem → residual **dilated** conv tower
with **squeeze-excite** → **attention pooling** → **reverse-complement-equivariant** read-out,
trained as a 3-seed ensemble with multi-task heads (activity regression + is-active + is-silencer),
optionally with one **non-B DNA** (G4Hunter) input channel. Best cortex model:
`activity_mt` with the i-motif/G4 channel + silencer head.

## Repository layout

```
src/                 all pipeline code (see "Scripts" below)
report/final_report.pdf
RESULTS_SUMMARY.xlsx consolidated result tables (11 sheets)
results/
  tables/            per-analysis CSVs (variant priority, saturation, TOMTOM, motifs, designs, …)
  figures/           key figures + saturation_maps/ (per-candidate ISM maps)
  motifs/            learned + JASPAR motifs in MEME format (for Tomtom)
  designs/           synthetic enhancer sequences (FASTA) + activity table
data/README.md       how to obtain the input data (not committed; see below)
```

## Scripts (`src/`)

| Script | Purpose |
|---|---|
| `common.py`, `models_best.py` | shared utils; the RegNetDNA architecture |
| `train_activity.py` | baseline activity model (ensemble, RC-averaged) |
| `train_activity_mt.py` | multi-task activity + is-active + silencer, non-B channels, `--assays`/`--extra`, `--warm-start` |
| `train_variant_best.py`, `train_variant_ens.py` | Siamese variant-effect model (single / ensemble) |
| `ism.py`, `satmut.py` | in-silico saturation mutagenesis maps + quantitative summary |
| `prioritize.py` | rank variants by functional prob, join GWAS/eQTL/ATAC annotations |
| `motifs.py`, `jaspar_match.py`, `motif_enrich.py`, `export_meme.py`, `meme_tomtom.py` | motif extraction, JASPAR matching, enrichment, MEME export, official TOMTOM |
| `design.py` | synthetic enhancer design by in-silico directed evolution |
| `g4.py` | G4Hunter / i-motif / R-loop non-B DNA channels |
| `prepare_promoter.py`, `train_transfer.py`, `patch_warmstart.py` | multi-cell (WTC11) ingest + pretrain→fine-tune transfer |
| `baselines`, `make_report.py`, `build_excel.py` | k-mer baseline, report + workbook generators |

## Reproduce

```bash
pip install -r requirements.txt
# 1. obtain data (see data/README.md) -> data/activity_seq.csv, data/variants_alleles.csv
python src/train_activity_mt.py --activity data/activity_seq.csv --imotif \
    --epochs 60 --seeds 3 --channels 192 --blocks 6 --batch 512 --outdir results
python src/train_variant_ens.py --variants data/variants_alleles.csv \
    --warm-start models/activity_best.pt --epochs 50 --outdir results
python src/satmut.py --model models/activity_best.pt --candidates results/top_candidates.csv --topn 30
python src/meme_tomtom.py   # official TOMTOM vs JASPAR
```
Trained weights (`*.pt`) are not committed (large binaries); regenerate with the commands above.

## Data & citation

Input data is **not redistributed** here — see `data/README.md` for download links. Please cite:
Song, Pollard, Ahituv *et al.*, *Science* (doi:10.1101/2023.02.15.528663 / `adh0559`);
and for the WTC11 lentiMPRA, Agarwal *et al.*, *Nature* (2024).

## License

Code released for research use. Input data and JASPAR/MEME resources retain their original licenses.
