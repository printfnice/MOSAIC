# MOSAIC

MOSAIC is a reproducibility package for trustworthy multimodal single-cell annotation under donor shift, incomplete protein panels, leave-class-out rejection stress tests and quantitative interpretability audits.

This repository accompanies the MOSAIC manuscript. It contains lightweight source code, configuration files, split manifests, compact evidence tables, a tagged P1/seed-42 checkpoint, a ten-cell CPU audit demo, manuscript PDFs and LaTeX sources. Raw participant data, cached matrices and full prediction files are intentionally excluded from GitHub.

## Repository Contents

- `code/`: release source code and audit scripts.
- `config/`: dataset and model configuration files.
- `manifest/`: split manifests for nested leave-one-donor-out and leave-class-out protocols.
- `evidence/`: compact CSV evidence tables used by the manuscript.
- `testdata/`: synthetic smoke-test fixture; this is not scientific evidence.
- `manuscript/`: current manuscript PDF, supplementary PDF, LaTeX sources and selected figure/table assets.
- `checkpoint/`: one P1/seed-42 MOSAIC checkpoint and its sanitized configuration.
- `demo/`: ten preprocessed demonstration cells and a runnable audit-record generator.
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
- Current 58-label representative marker and CD8 boundary audit: `evidence/results/tables/mosaic_n_v8_marker_mechanism_audit_2026-08-14.csv`, `evidence/results/tables/mosaic_n_v8_cd8_mechanism_audit_2026-08-14.csv`
- Checksum-linked cell-level audit cases: `evidence/results/tables/mosaic_n_v8_audit_case_study_records_2026-08-14.csv` and Figure 3 assets under `manuscript/Fig/`

## Data And Checkpoints

Raw datasets are available from the public accessions described in the manuscript. GitHub does not host raw participant data, large cached tensors or full prediction files. The repository does include one P1/seed-42 checkpoint (`checkpoint/model.pt`, about 10 MB) and a CPU demo that emits an audit record from ten preprocessed demonstration cells. The checkpoint and demo are part of the tagged GitHub snapshot; no Zenodo DOI is claimed.

## Manuscript

- Main manuscript: `manuscript/MOSAIC.pdf`
- Supplementary material: `manuscript/MOSAIC_supplementary.pdf`
- LaTeX sources: `manuscript/main.tex`, `manuscript/supplementary.tex`, `manuscript/references.bib`

## Citation

For the software snapshot, cite the MOSAIC manuscript and the tagged GitHub snapshot `v7.0.1-submission`. No archival DOI is assigned to this snapshot.
