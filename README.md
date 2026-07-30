# MOSAIC

MOSAIC is a reproducibility package for trustworthy multimodal single-cell annotation under donor shift, incomplete protein panels, leave-class-out rejection stress tests and quantitative interpretability audits.

This repository accompanies the MOSAIC manuscript. It contains lightweight source code, configuration files, split manifests, compact evidence tables, synthetic smoke-test data, manuscript PDFs and LaTeX sources. Raw participant data, cached matrices, full prediction files and model checkpoints are intentionally excluded from GitHub.

## Repository Contents

- `code/`: release source code and audit scripts.
- `config/`: dataset and model configuration files.
- `manifest/`: split manifests for nested leave-one-donor-out and leave-class-out protocols.
- `evidence/`: compact CSV evidence tables used by the manuscript.
- `testdata/`: synthetic smoke-test fixture; this is not scientific evidence.
- `manuscript/`: current manuscript PDF, supplementary PDF, LaTeX sources and selected figure/table assets.
- `release_manifest.csv`: file-level source manifest with SHA-256 hashes.
- `CHECKSUMS.sha256`: checksums for packaged files.
- `commands.txt`: main reproduction commands used for the evidence package.

## Quick Smoke Test

From the repository root:

```bash
python -m pip install -r requirements.txt
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python code/experiments/reproducibility/v33_release_smoke_test.py --release-dir .
```

The smoke test validates package structure and a small synthetic fixture. It does not reproduce the scientific experiments.

Optional lightweight tests:

```bash
python -m pytest code/experiments -q
```

## Main Evidence

The primary PBMC 3P experiment uses nested leave-one-donor-out evaluation across eight donors and 58 fine-level labels. Compact evidence tables include donor-level MOSAIC/MLP summaries, checkpoint ablations, missing-panel stress tests, leave-class-out rejection, PDC101 comparison, attribution stability, CellTypist baseline results and MOSAIC-versus-XGBoost donor-paired statistics.

Recent manuscript-facing additions:

- CellTypist full-training published baseline: `evidence/results/tables/mosaic_n_v42_celltypist_pbmc_nested_donor_2026-07-29.csv`
- Full-training published and strong ML baseline table: `evidence/results/tables/mosaic_n_v37_strong_ml_expanded_baselines_2026-07-29.csv`
- MOSAIC full versus Early-fusion XGBoost paired statistics: `evidence/results/tables/mosaic_n_v43_mosaic_vs_xgboost_paired_2026-07-29.csv`

## Data And Checkpoints

Raw datasets are available from the public accessions described in the manuscript. GitHub is not used to host raw participant data, large cached tensors or full model checkpoints.

Before journal submission, checkpoint hosting should be added through a stable archive such as Zenodo or GitHub Releases with Git LFS. The DOI/checkpoint URL should then be inserted into the manuscript Code availability section.

## Manuscript

- Main manuscript: `manuscript/MOSAIC.pdf`
- Supplementary material: `manuscript/MOSAIC_supplementary.pdf`
- LaTeX sources: `manuscript/main.tex`, `manuscript/supplementary.tex`, `manuscript/references.bib`

## Citation

A formal citation and archival DOI will be added after the public release is finalized.
