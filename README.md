# MOSAIC V8.2 submission candidate

MOSAIC is a reproducibility package for trustworthy multimodal single-cell annotation under donor shift, incomplete protein panels, leave-class-out rejection stress tests and quantitative interpretability audits.

This local package accompanies the V8.2 MOSAIC manuscript candidate. It contains lightweight source code, configuration files, split manifests, compact evidence tables, the V8.2 closure audit artifacts, a ten-cell CPU audit demo, manuscript PDFs and LaTeX sources. Raw participant data, cached matrices, full prediction files and checkpoints are intentionally excluded.

## Repository Contents

- `code/`: release source code and audit scripts.
- `config/`: dataset and model configuration files.
- `manifest/`: split manifests for nested leave-one-donor-out and leave-class-out protocols.
- `evidence/`: compact CSV evidence tables used by the manuscript.
- `evidence/results/experiments/v8.2_missing_modality_pdc_audit/`: V8.2 size-matched null, PDC101 paired audit and same-study 5P transfer artifacts.
- `testdata/`: synthetic smoke-test fixture; this is not scientific evidence.
- `manuscript/`: current manuscript PDF, supplementary PDF, LaTeX sources and selected figure/table assets.
- `demo/`: ten preprocessed demonstration cells, an input manifest, and a runnable checkpoint-free audit-validation demo.
- `MOSAIC_DATASET_PROVENANCE.md`: accession/version, training/evaluation role, split unit, preprocessing and ground-truth annotation record for every dataset used in the manuscript.
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

## Reproducible Demo On Packaged Test Data

The released test-data link is [`demo/inputs.npz`](demo/inputs.npz), with the
cell-level schema documented in [`demo/input_manifest.csv`](demo/input_manifest.csv).
It contains ten preprocessed demonstration cells with a `gene` matrix of shape
`10 x 3000`, a `protein` matrix of shape `10 x 224`, and stable `cell_id`
values. The fixture is included for installation, input/output, and audit-schema
verification; it is not a raw participant dataset and carries no biological
ground-truth claim.

From the repository root, run:

```bash
python -m pip install -r requirements.txt
python demo/run_demo.py --package . --output-dir demo/output
```

The command reads `demo/inputs.npz` and the packaged audit record
`demo/audit_record_generated.csv`. It writes:

- `demo/output/audit_record_validated.csv`: the validated ten-row audit record,
  including final and branch predictions, confidence/uncertainty, margins,
  modality weights, branch conflict, HSR gate and HSR delta norm.
- `demo/output/demo_summary.json`: device, input shapes, cell count, output path,
  and the explicit `checkpoint_required` and `scientific_inference` flags.

This is a checkpoint-free audit-validation demonstration. It does not run
trained-model inference, train a model, or reproduce manuscript performance.
The public package deliberately excludes checkpoints and raw participant data.

## Dataset Provenance

See [`MOSAIC_DATASET_PROVENANCE.md`](MOSAIC_DATASET_PROVENANCE.md) for the source
accession or repository, version identifier, training/evaluation role, split
unit, preprocessing boundary, and source/process of ground-truth annotations for
the PBMC, E-MTAB-10026, COMBAT/COVID, and GSE229791 datasets. The document also
separates biological evaluation datasets from the packaged ten-cell demo fixture
and states which files are excluded from this release.

## Main Evidence

The primary PBMC 3P experiment uses nested leave-one-donor-out evaluation across eight donors and 58 fine-level labels. Compact evidence tables include donor-level MOSAIC/MLP summaries, inference ablations, missing-panel stress tests, leave-class-out rejection, PDC101 comparison, attribution stability, CellTypist baseline results and MOSAIC-versus-XGBoost donor-paired statistics.

Recent manuscript-facing additions:

- CellTypist full-training published baseline: `evidence/results/tables/mosaic_n_v42_celltypist_pbmc_nested_donor_2026-07-29.csv`
- Full-training published and strong ML baseline table: `evidence/results/tables/mosaic_n_v37_strong_ml_expanded_baselines_2026-07-29.csv`
- MOSAIC full versus Early-fusion XGBoost paired statistics: `evidence/results/tables/mosaic_n_v43_mosaic_vs_xgboost_paired_2026-07-29.csv`
- Current 58-label representative marker and CD8 boundary audit: `evidence/results/tables/mosaic_n_v8_marker_mechanism_audit_2026-08-14.csv`, `evidence/results/tables/mosaic_n_v8_cd8_mechanism_audit_2026-08-14.csv`
- Checksum-linked cell-level audit cases: `evidence/results/tables/mosaic_n_v8_audit_case_study_records_2026-08-14.csv` and Figure 3 assets under `manuscript/Fig/`
- V8.2 size-matched random ADT null: `evidence/results/experiments/v8.2_missing_modality_pdc_audit/pbmc_random_adt_null/`
- V8.2 PDC101 paired audit: `evidence/results/experiments/v8.2_missing_modality_pdc_audit/pdc101_paired_audit/`
- V8.2 same-study 5P transfer audit: `evidence/results/experiments/v8.2_missing_modality_pdc_audit/secondary_cohort/`

## Data And Artifacts

Raw datasets are available from the public accessions described in
`MOSAIC_DATASET_PROVENANCE.md`. This package does not host raw participant data,
large cached tensors, full prediction files or checkpoints. The CPU demo reads
the packaged test input and writes a validated audit record plus summary; it
checks schema and consistency without performing model inference or reproducing
the full scientific training runs. No Zenodo DOI or public V8.2 tag is claimed
by this local candidate.

## Manuscript

- Main manuscript: `manuscript/MOSAIC.pdf`
- Supplementary material: `manuscript/MOSAIC_supplementary.pdf`
- LaTeX sources: `manuscript/main.tex`, `manuscript/supplementary.tex`, `manuscript/references.bib`

## Citation

For the software snapshot, cite the MOSAIC manuscript and the verified public GitHub snapshot `v8.1.4-submission`. The V8.2 closure artifacts are included in this local submission candidate; no archival DOI or public V8.2 tag is assigned here.
