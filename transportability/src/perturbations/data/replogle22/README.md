# Replogle22

This directory builds three K562-anchored perturbation datasets:

- `K562 + RPE1`
- `K562 + Jurkat`
- `K562 + HepG2`

## Download raw data

Raw inputs are downloaded separately from preprocessing:

```bash
python -m data.replogle22.download_data
```

The downloader validates existing files and skips valid ones. Select specific files or replace valid files with:

```bash
python -m data.replogle22.download_data --datasets K562 Jurkat
python -m data.replogle22.download_data --datasets K562 --force
```

K562 and RPE1 come from Zenodo record `13350497`. Jurkat and HepG2 come from GEO accession `GSE264667`.

## Preprocess data

Generate all three pairs with the default settings:

```bash
python -m data.replogle22.get_data
```

Generate selected pairs or override preprocessing parameters:

```bash
python -m data.replogle22.get_data --partners RPE1 Jurkat
python -m data.replogle22.get_data \
  --min-cells-per-pert-per-cell-line 64 \
  --min-genes-per-cell 200 \
  --min-cells-per-gene 3 \
  --normalize-target-sum 10000 \
  --hvg-top-genes 8192
```

Outputs are written to `RPE1/`, `Jurkat/`, and `HepG2/`. Each selected directory contains:

- `processed.h5ad`
- `genes.csv.gz`
- `names_df_vsrest.pkl`
- `scores_df_vsrest.pkl`
- `names_df_vsctrl.pkl`
- `scores_df_vsctrl.pkl`
- `processed_manifest.json`
