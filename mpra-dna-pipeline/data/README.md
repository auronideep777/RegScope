# Data

Input data is not committed to this repo. Obtain it as follows and place under `data/`.

## Cortex MPRA (Song / Pollard / Ahituv, Science adh0559)
- Supplement `science.adh0559_data_s1_to_s3.zip`:
  - **S1** (`adh0559_data_s1.xlsx`, sheets Primary/Organoids) -> insert-level activity.
  - **S2** (nested `adh0559_data_s2.zip` -> `DataS2-Variant-library-ratios.xlsx`) -> variant library.
- `src/prepare_data.py` (or the provided ingest) converts these to:
  - `data/activity_seq.csv`  (insert_name, chrom, start, end, assay, log2_ratio, is_active, is_silencer, sequence)
  - `data/variants_alleles.csv` (rsid, chrom, pos, ref, alt, assay, logFC, adj_p, ratios, annotations, sequence, var_offset)

Provenance verified: these tables are byte-identical to S1/S2 (activity values and logFC match to 1e-16).

## WTC11 lentiMPRA (Agarwal et al., for cross-cell experiments)
- Zenodo record 10558183 -> `human_legnet-main/datasets/original/WTC11_averaged.tsv`
  (columns: seq_id, seq, mean_value, fold, rev).
- Convert with: `python src/prepare_promoter.py --tsv WTC11_averaged.tsv --assay WTC11 --outdir data`

## Reference resources (auto-downloaded by scripts)
- JASPAR2024 CORE vertebrates via `pyjaspar` (offline bundled).
