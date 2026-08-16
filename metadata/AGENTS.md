# MOSAIC Release Metadata

This directory stores lightweight metadata for the public MOSAIC reproducibility
package. It intentionally excludes local agent operating notes, raw datasets,
cache directories, trained checkpoints and machine-specific runtime state.

Release scope:

- Source code and audit scripts needed to inspect the reported experiments.
- Dataset configuration files and split manifests.
- Compact manuscript-facing evidence tables.
- Synthetic smoke-test data for package validation only.
- Manuscript PDFs, LaTeX sources and selected figure/table assets.
- A ten-cell CPU audit-record validation demo that does not require inference; checkpoints are intentionally excluded.
- File manifest and SHA-256 checksum records.

Excluded by design:

- Raw participant-level single-cell matrices.
- Large cached feature arrays or prediction dumps.
- Raw participant-level single-cell matrices, large cached feature arrays, full prediction dumps, training-only artifacts and checkpoints.
- Local conda environments, machine paths and private credentials.

The GitHub repository is intended as a lightweight source and evidence package.
The `v8.1.4-submission` snapshot contains the CPU audit demo and compact evidence
tables; no checkpoint, Zenodo DOI or raw-data archive is claimed for this release.
