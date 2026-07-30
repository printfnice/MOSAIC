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
- File manifest and SHA-256 checksum records.

Excluded by design:

- Raw participant-level single-cell matrices.
- Large cached feature arrays or prediction dumps.
- Model checkpoints.
- Local conda environments, machine paths and private credentials.

The GitHub repository is intended as a lightweight source and evidence package.
Large reproducibility artifacts such as checkpoints should be archived through a
stable external service, for example Zenodo or GitHub Releases, if they are
needed for a formal journal submission.
