# Cover Letter

Dear Editors,

Please consider our manuscript, **“MOSAIC: auditable RNA–ADT annotation with branch-level modality evidence under donor shift and incomplete protein panels,”** for publication as an Original Paper in *Bioinformatics*.

MOSAIC addresses a practical limitation of multimodal single-cell annotation: a predicted label alone does not show whether RNA and protein evidence agree, whether a local hierarchy correction was applied, or when a prediction should be reviewed. The method therefore emits modality-specific labels and margins, a bounded sibling-level refinement action, and a cell-level audit record alongside the final annotation.

The primary evidence uses nested leave-one-donor-out evaluation of 161,764 PBMC cells and 58 fine-level labels. Relative to a loss- and selection-matched early-fusion MLP, MOSAIC improves macro-F1 and balanced accuracy, while the aggregate ACC and weighted-F1 intervals cross zero. More importantly, the branch-aware audit score improves error-detection AUROC from 0.6914 to 0.8401 and reduces accepted-cell error risk at 50% coverage from 0.0565 to 0.0169. Panel masking, leave-class-out stress tests, attribution audits and an external hierarchy comparison identify concrete failure regimes rather than treating a single accuracy value as sufficient evidence.

The manuscript is deliberately protocol-defined. It does not claim universal SOTA, arbitrary missing-modality robustness or solved open-set recognition. The size-matched six- and 15-ADT null, an exact paired PDC101 audit and a same-study GSE164378 5P transfer audit are now included with explicit scope caveats. Code, configurations, compact evidence artifacts, tests and a checkpoint-free CPU audit validator are publicly available under the MIT License in the verified `v8.1.4-submission` snapshot; the V8.2 closure summaries and source paths are included in the submission package. Raw data, caches, full predictions and checkpoints are excluded because of size and data-distribution constraints.

The authors declare no specific funding and no competing interests. Author contributions and the detailed use of AI-assisted tools are stated in the manuscript and Supplementary Material. The corresponding author is Linhao Wu (2544814855@qq.com).

Sincerely,

Linhao Wu, on behalf of all authors
