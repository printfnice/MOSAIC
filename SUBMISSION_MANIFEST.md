# MOSAIC v8.2 Submission Manifest

## Manuscript files

- `MOSAIC.pdf`: main manuscript, generated from the current `main.tex`.
- `MOSAIC_DATASET_PROVENANCE.md`: source, version, split, and ground-truth annotation record for the datasets used in the manuscript.

## Reproducibility entry points

- `demo/`: packaged test inputs, input manifest, and a runnable checkpoint-free audit-validation demo that writes explicit CSV and JSON outputs.
- `README.md`: installation, test-data command, input/output schema, and scientific-scope boundary.
- `MOSAIC_supplementary.pdf`: supplementary material, generated from the current `supplementary.tex`.
- `FIGURE_ALT_TEXT.md`: figure accessibility descriptions for the submission form.
- `COVER_LETTER.md`: cover-letter draft for the online submission form.

Current PDF hashes:

- `MOSAIC.pdf`: `e93067919bbb44eed6a9585eab2b99dd84677af420fc2c15736e6fd1371dd779`
- `MOSAIC_supplementary.pdf`: `bed88ab0496b48a61ede87cca73533027743967260260c5ec55cc1f185aa216e`

## Tagged software snapshot

- Repository: `https://github.com/printfnice/MOSAIC`
- Branch/tag: `v8.1.4-submission` (verified public code snapshot; V8.2 is a local submission candidate)
- Published package commit: `0a782b2844833d32454be68c56e738ffcc71f64b`
- Current access check: anonymous HTTPS verification of `v8.1.4-submission` passed; SSH deploy-key access is unavailable on the current machine (`Permission denied (publickey)`).
- License: MIT
- Included: source code, configurations, compact evidence artifacts, tests, checksums, manuscript sources and a checkpoint-free ten-cell CPU audit validator. The current submission source additionally records the V8.2 closure artifacts.
- Excluded: raw participant data, caches, full prediction files and checkpoints.

## Scientific scope

The manuscript reports protocol-defined donor-disjoint PBMC evidence and now includes the V8.2 size-matched ADT null, exact PDC101 paired boundary audit and same-study GSE164378 5P transfer audit. The size-matched result remains a fixed-panel frozen-checkpoint diagnostic; the PDC101 paired difference is non-significant (`P=0.687149`); and the 5P transfer is not independent external validation because it shares the study and P1--P8 donors. An immutable archival DOI, raw-data archive and public V8.2 tag are not claimed by this local candidate.

## V8.2 evidence artifacts

- `results/experiments/v8.2_missing_modality_pdc_audit/pbmc_random_adt_null/`: 24 frozen donor--seed units with six- and 15-ADT size-matched random nulls.
- `results/experiments/v8.2_missing_modality_pdc_audit/pdc101_paired_audit/`: 2,098 exact cell-ID pairs and paired McNemar audit.
- `results/experiments/v8.2_missing_modality_pdc_audit/secondary_cohort/`: 49,147-cell same-study 5P assay-cohort transfer audit.
- `manufacture/mosaic_n_bioinformatics_manuscript_v1/oup-authoring-template/tables/v82/supplement_v82_audit_closure.tex`: manuscript-facing U4 summary fragment.

SHA-256 values are regenerated after the final LaTeX build and release-package assembly.
