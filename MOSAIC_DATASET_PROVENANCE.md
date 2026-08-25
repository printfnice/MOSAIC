# MOSAIC Dataset Provenance

This file documents every dataset used in the manuscript. Raw participant data are not redistributed in this repository. Each scientific dataset is versioned by its public accession, the cited source publication or release, and the project split/preprocessing records. `CHECKSUMS.sha256` covers packaged compact artifacts only; it does not claim checksums for excluded raw files.

| Dataset | Public source/version | Role | Size and labels | Ground-truth / reference labels | Split and caveat |
|---|---|---|---|---|---|
| PBMC 3P | GEO `GSE164378`; Hao et al., *Cell* 2021; DOI `10.1016/j.cell.2021.04.048`; https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE164378 | Primary donor-disjoint experiment | 161,764 cells, 8 donors, 224 ADTs, 58 L3 labels | Source WNN integration/clustering plus manual annotation; the source label space, including `Doublet`, is retained | Nested leave-one-donor-out; donor is the statistical unit; feature selection and normalization are training-donor-only |
| PBMC 5P | The 5P assay cohort from the same `GSE164378` study | Secondary panel/breadth evidence | 49,147 cells, 54 proteins | Source WNN/manual annotations | Shares the study and donors with 3P; not independent external validation |
| E-MTAB-10026 | ArrayExpress accession `E-MTAB-10026`; Stephenson et al., *Nature Medicine* 2021; https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-10026 | Patient-disjoint breadth and panel-mismatch audit | 211,150 exact-compatible cells, 130 patients, 11 labels | Study-provided cell-type annotations, used only for final evaluation/error definition | Patient-disjoint split; PBMC audit coefficients are transferred without target-label fitting |
| COMBAT/COVID shift | COMBAT Consortium, *Cell* 2022, DOI `10.1016/j.cell.2022.01.012`; Zenodo `10.5281/zenodo.6120249`; HCA project `cdabcf0b-7602-4abf-9afb-3b410e545703`; EGA `EGAS00001005493` | Parent-safe disease-shift breadth evaluation | Source-specific aligned label space | COMBAT source annotations and disease labels | Labels are evaluation-only; cross-method consensus is not attributed to one MOSAIC model |
| GSE229791 PDC101 | GEO accession `GSE229791`; Caron et al., *Nature Immunology* 2025; https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE229791 | External hierarchy/sorted boundary benchmark | 2,098-cell holdout; official MMoCHi hierarchy | Official sorted labels parsed from HTO/sort metadata; HTO is excluded from model inputs | `external_holdout=True`; isotype/control proteins excluded; reported separately from PBMC L3 |
| Demo fixture | Packaged `demo/inputs.npz` and `demo/input_manifest.csv` | Installation/schema/audit demo, not scientific evidence | 10 cells, 3,000 RNA and 224 ADT features | No biological ground truth is claimed | CPU-only checkpoint-free audit-record validation |

## Preprocessing and annotation boundaries

- PBMC RNA/ADT preprocessing is fitted on training donors within each fold. E-MTAB transfer uses fixed PBMC-derived audit coefficients and does not fit on target labels. Missing-channel fill controls are explicit protocol variants.
- PDC101 HTO values are used only to recover the official sorted benchmark label; they are never model features. This prevents label leakage.
- A ground-truth label means the source study annotation or sorted benchmark metadata used for evaluation, not an independent biological gold standard. PBMC L3 labels derive from WNN clustering and manual annotation; PDC101 labels derive from the official sorted benchmark; COVID/COMBAT/E-MTAB labels are inherited from their source studies.
- Scientific raw matrices, caches, full predictions and checkpoints are excluded. Packaged source files, compact evidence, demo inputs and manuscript assets are listed in `release_manifest.csv` and checked by `CHECKSUMS.sha256`.

