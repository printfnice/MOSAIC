# MOSAIC Figure Alt Text

These descriptions are prepared for the Bioinformatics submission form and are separate from the figure captions.

## Figure 1

**Short description:** MOSAIC processes paired RNA and ADT inputs through separate encoders, multimodal fusion, hierarchical prediction, bounded sibling refinement and cell-level audit output.

**Long description:** A left-to-right architecture diagram shows paired RNA and ADT matrices entering separate modality encoders. The two representations are fused and passed to RNA, ADT and fused prediction branches. Margin-based arbitration selects the evidence-emitting path, and a bounded hierarchy-aware sibling refinement module can modify logits only within declared local sibling groups. The final output includes the predicted fine-level label, confidence, branch labels and margins, fusion information, refinement action and feature-attribution fields.

## Figure 2

**Short description:** Six panels quantify donor-disjoint performance, module comparisons, audit scores, branch margins, missing-protein sensitivity and attribution evidence.

**Long description:** Panel a shows donor-paired absolute scores for MOSAIC and the matched MLP. Panel b shows retrained branch-loss variants and the locked evidence-emitting configuration. Panel c shows error-detection AUROC for uncertainty and branch-derived audit scores. Panel d relates branch top-two margins to branch accuracy across margin groups. Panel e shows performance and calibration changes under deterministic protein-panel masks. Panel f shows attribution deletion lift and canonical-marker overlap across selected sibling classes and modalities.

## Figure 3

**Short description:** Four scatter plots relate per-class PBMC macro-F1 to representative ADT-marker hits, RNA-marker hits, total marker hits and mean fusion weight across 58 labels.

**Long description:** Each point represents one PBMC fine-level label, with point size indicating held-out support and color indicating whether a non-empty representative ADT marker list is fully present in the panel. The four panels compare per-class macro-F1 with ADT marker hits, RNA marker hits, total marker hits and mean fusion weight. The descriptive associations are near zero, showing that marker availability and model feature use are distinct audit fields.

## Supplementary Figure S1

**Short description:** Risk-coverage curves compare uncertainty and branch-evidence scores for donor-held-out error triage.

**Long description:** The plot reports held-out error risk among accepted cells as accepted coverage increases. Curves compare final uncertainty, binary branch conflict and the combined donor-held-out uncertainty--branch score. The combined score lowers risk in the low-coverage, high-precision region and converges with final uncertainty near 90 percent coverage.

## Supplementary marker-limit case study figure

**Short description:** Two case studies show panel-limited leave-class-out rejection and a PDC101 CD8 boundary comparison.

**Long description:** Panel a summarizes the B-naive-lambda leave-class-out failure, including weak unknown detection, limited parent-safe fallback and the absence of a matching kappa/lambda-like ADT marker. Panel b compares class-level F1 differences between MOSAIC and MMoCHi on the sorted PDC101 holdout, with the largest gaps concentrated in CD8 memory and effector-memory-related classes.
