#!/usr/bin/env python
"""Build a current-checkpoint 58-label marker-evidence audit.

The output is intentionally descriptive.  The marker map records representative
panel evidence for a label family; it is not a claim that one antibody defines a
cell state or that attribution is causal.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
V33 = ROOT / "experiments/exp_generalization/mosaic_n_v33"
if str(V33) not in sys.path:
    sys.path.insert(0, str(V33))

from evaluate_v33_checkpoint_ablations import DONORS, SEEDS, load_checkpoint  # noqa: E402


DATE = "2026-08-14"
METHOD = "mosaic_full"

# These are representative panel tokens, deliberately narrower than a complete
# biological marker ontology.  They are used only for an availability audit.
ADT_GROUPS = {
    "b_cell": ["CD19", "CD20", "CD79a", "CD79b", "HLA-DR"],
    "cd4": ["CD3", "CD4"],
    "cd8": ["CD3", "CD8"],
    "mono": ["CD14", "CD16", "CD64", "CD11b", "HLA-DR"],
    "nk": ["CD56", "CD335", "CD16", "CD244"],
    "dc": ["CD1c", "CD11c", "HLA-DR"],
    "pdc": ["CD123", "CD303", "CD304", "HLA-DR"],
    "t": ["CD3", "CD2"],
    "gd_t": ["CD3", "TCR-V-2", "TCR-V-9"],
    "mait": ["CD3", "CD8", "CD161", "TCR-V-24-J-18"],
    "plasma": ["CD38", "CD138", "CD319"],
    "platelet": ["CD41", "CD61", "CD42b"],
    "eryth": ["CD235a", "CD71"],
    "hspc": ["CD34", "CD117", "CD133"],
    "treg": ["CD3", "CD4", "CD25", "CD127", "CD152"],
    "ilc": ["CD127", "CD294"],
}
RNA_GROUPS = {
    "b_cell": ["MS4A1", "CD79A", "CD79B", "CD37"],
    "cd4": ["CD3D", "CD3E", "IL7R", "LTB"],
    "cd8": ["CD3D", "CD3E", "CD8A", "CD8B"],
    "mono": ["LYZ", "S100A8", "S100A9", "FCN1", "CTSS"],
    "nk": ["NKG7", "GNLY", "KLRD1", "TRBC2"],
    "dc": ["FCER1A", "CST3", "CLEC10A"],
    "pdc": ["GZMB", "GZMB", "IRF7", "GZMB"],
    "t": ["CD3D", "CD3E", "TRBC2"],
    "gd_t": ["TRDC", "TRGC1", "TRGC2", "CD3D"],
    "mait": ["KLRB1", "SLC4A10", "TRAV1-2", "CD3D"],
    "plasma": ["MZB1", "JCHAIN", "SEC11C", "CD79A"],
    "platelet": ["PPBP", "PF4", "NRGN"],
    "eryth": ["HBB", "HBA1", "HBA2", "ALAS2"],
    "hspc": ["MPO", "GATA2", "CD34", "HLF"],
    "treg": ["FOXP3", "IL7R", "IL32", "CTLA4"],
    "ilc": ["IL7R", "KLRB1", "RORA"],
}


def normalize_feature(value: str) -> str:
    text = str(value).upper().replace("_", "-").strip()
    for suffix in ("-1", "-2"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    if text == "CD8A":
        return "CD8"
    if text == "CD4A":
        return "CD4"
    return text


def label_group(label: str) -> str | None:
    text = str(label)
    if text == "Doublet" or "Proliferating" in text:
        return None
    if text.startswith("B "):
        return "b_cell"
    if text.startswith("CD4") or text.startswith("Treg"):
        return "treg" if text.startswith("Treg") else "cd4"
    if text.startswith("CD8"):
        return "cd8"
    if "Mono" in text:
        return "mono"
    if text.startswith("NK"):
        return "nk"
    if text.startswith("cDC"):
        return "dc"
    if text == "pDC":
        return "pdc"
    if text.startswith("gdT"):
        return "gd_t"
    if text == "MAIT":
        return "mait"
    if text in {"Plasma", "Plasmablast"}:
        return "plasma"
    if text == "Platelet":
        return "platelet"
    if text == "Eryth":
        return "eryth"
    if text == "HSPC":
        return "hspc"
    if text == "ILC":
        return "ilc"
    if text == "T":
        return "t"
    return None


def representative_markers(label: str, modality: str) -> list[str]:
    group = label_group(label)
    if group is None:
        return []
    source = ADT_GROUPS if modality == "ADT" else RNA_GROUPS
    markers = list(source.get(group, []))
    text = str(label)
    if modality == "ADT" and text.startswith("B "):
        if "naive" in text.lower():
            markers += ["IgD", "IgM"]
        elif "memory" in text.lower():
            markers += ["CD27", "CD45RO"]
        # The current panel has no kappa/lambda surface feature.
    if modality == "ADT" and "TCM" in text:
        markers += ["CD45RO", "CD27", "CD127"]
    if modality == "ADT" and "TEM" in text:
        markers += ["CD45RO", "CD95", "CD57"]
    if modality == "ADT" and "Naive" in text:
        markers += ["CD45RA", "CD27", "CD127"]
    return list(dict.fromkeys(markers))


def select_correct_indices(
    model: torch.nn.Module,
    arrays: dict,
    class_names: np.ndarray,
    max_per_class: int,
    seed: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    indices = np.asarray(arrays["test_idx"], dtype=np.int64)
    gene = torch.as_tensor(arrays["gene"][indices], dtype=torch.float32, device=device)
    protein = torch.as_tensor(arrays["protein"][indices], dtype=torch.float32, device=device)
    labels = np.asarray(arrays["labels"][indices], dtype=np.int64)
    with torch.no_grad():
        outputs = model(
            gene,
            protein,
            availability_mask=torch.ones(len(indices), 2, device=device),
        )
        predicted = outputs["final_logits"].argmax(dim=1).detach().cpu().numpy()
    rng = np.random.default_rng(seed)
    selected: dict[str, np.ndarray] = {}
    for class_index, class_label in enumerate(class_names):
        candidates = indices[(labels == class_index) & (predicted == class_index)]
        if len(candidates) > max_per_class:
            candidates = np.sort(rng.choice(candidates, max_per_class, replace=False))
        selected[str(class_label)] = np.asarray(candidates, dtype=np.int64)
    return selected


def attribution_batch(
    model: torch.nn.Module,
    gene: torch.Tensor,
    protein: torch.Tensor,
    target_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    gene = gene.detach().clone().requires_grad_(True)
    protein = protein.detach().clone().requires_grad_(True)
    outputs = model(
        gene,
        protein,
        availability_mask=torch.ones(len(gene), 2, device=gene.device),
    )
    target = outputs["final_logits"][:, target_index].sum()
    grad_gene, grad_protein = torch.autograd.grad(target, (gene, protein))
    return (
        (grad_gene * gene).detach().abs().cpu().numpy(),
        (grad_protein * protein).detach().abs().cpu().numpy(),
    )


def load_f1_and_weights() -> tuple[pd.DataFrame, pd.DataFrame]:
    f1 = pd.read_csv(
        ROOT / "results/exp_generalization/mosaic_n_v33/donor_summary/per_class_method_summary.csv"
    )
    f1 = f1[f1["method"].eq(METHOD)].copy()
    if f1["class_label"].nunique() != 58:
        raise ValueError("current MOSAIC per-class summary does not cover 58 labels")
    rows = []
    for donor in DONORS:
        for seed in SEEDS:
            path = (
                ROOT
                / "results/exp_generalization/mosaic_n_v33/donor_matrix"
                / f"test_{donor}"
                / f"mosaic_full_seed{seed}"
                / "weight_summary_by_class.csv"
            )
            if not path.is_file():
                raise FileNotFoundError(path)
            frame = pd.read_csv(path)
            frame["test_donor"] = donor
            frame["seed"] = seed
            rows.append(frame)
    weights = pd.concat(rows, ignore_index=True)
    return f1, weights


def tex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
        "#": r"\#",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def write_manuscript_fragments(summary: pd.DataFrame, targets: pd.DataFrame) -> None:
    """Write compact supplementary diagnostics from the audited CSVs."""
    table_dir = ROOT / "manufacture/mosaic_n_bioinformatics_manuscript_v1/oup-authoring-template/tables/v7"
    table_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for label in sorted(summary["class_label"].unique()):
        adt = summary[(summary["class_label"].eq(label)) & summary["modality"].eq("ADT")].iloc[0]
        rna = summary[(summary["class_label"].eq(label)) & summary["modality"].eq("RNA")].iloc[0]
        group = tex_escape(adt["marker_group"] if adt["marker_group"] != "none" else "none")
        available = f"{int(adt['n_available_panel_markers'])}/{int(adt['n_representative_markers'])}" if int(adt["n_representative_markers"]) else "--"
        rows.append(
            f"{tex_escape(label)} & {group} & {available} & {int(rna['top20_marker_hits'])} & {int(adt['top20_marker_hits'])} & {float(adt['mean_f1']):.4f} & {int(adt['total_test_support'])} & {float(adt['mean_fusion_weight']):.3f} \\\\" 
        )
    target_rows = []
    for _, row in targets.sort_values("target_label").iterrows():
        target_rows.append(
            f"{tex_escape(row['target_label'])} & {tex_escape('Y' if row['panel_marker_available'] else 'N')} & {float(row['mean_f1']):.4f} & {float(row['unknown_recall']):.4f} & {float(row['unknown_auroc']):.4f} \\\\"
        )
    pdc_path = ROOT / "results/tables/mosaic_n_v33_pdc101_mmochi_per_class_2026-07-23.csv"
    pdc = pd.read_csv(pdc_path)
    mapping = {"CD8 Naive": "cd8_n", "CD8 TCM_1": "cd8_cm", "CD8 TEM_4": "cd8_em"}
    cd8_rows = []
    cd8_summary = []
    for pbmc_label, pdc_label in mapping.items():
        adt = summary[(summary["class_label"].eq(pbmc_label)) & summary["modality"].eq("ADT")].iloc[0]
        mosaic = pdc[(pdc["class_label"].eq(pdc_label)) & pdc["method"].eq("MOSAIC-N")].iloc[0]
        mmochi = pdc[(pdc["class_label"].eq(pdc_label)) & pdc["method"].eq("MMoCHi")].iloc[0]
        # Keep displayed derived values internally consistent with the displayed components.
        mosaic_f1 = float(f"{float(mosaic['mean_f1']):.4f}")
        mmochi_f1 = float(f"{float(mmochi['mean_f1']):.4f}")
        delta = round(mmochi_f1 - mosaic_f1, 4)
        cd8_summary.append(
            {
                "pbmc_label": pbmc_label,
                "pdc101_class": pdc_label,
                "adt_available_representative": f"{int(adt['n_available_panel_markers'])}/{int(adt['n_representative_markers'])}",
                "adt_top20_marker_hits": int(adt["top20_marker_hits"]),
                "mean_fusion_weight": float(adt["mean_fusion_weight"]),
                "mosaic_f1": mosaic_f1,
                "mmochi_f1": mmochi_f1,
                "mmochi_minus_mosaic_f1": delta,
                "interpretation": "label-space and preprocessing aligned only at the holdout level; descriptive boundary comparison",
            }
        )
        cd8_rows.append(
            f"{tex_escape(pbmc_label)} & {tex_escape(pdc_label)} & {int(adt['n_available_panel_markers'])}/{int(adt['n_representative_markers'])} & {int(adt['top20_marker_hits'])} & {float(adt['mean_fusion_weight']):.3f} & {mosaic_f1:.4f} & {mmochi_f1:.4f} & {delta:.4f} \\\\"
        )
    cd8_path = ROOT / f"results/tables/mosaic_n_v8_cd8_mechanism_audit_{DATE}.csv"
    pd.DataFrame(cd8_summary).to_csv(cd8_path, index=False)
    mirror = ROOT / "output/tables"
    mirror.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cd8_summary).to_csv(mirror / cd8_path.name, index=False)
    content = r"""% Generated by build_v8_marker_mechanism_audit.py; the longtable consumes S13.
\noindent\textbf{S13. 58-label representative panel-evidence audit.}
The first table covers every current PBMC L3 label and reports the number of representative markers present in the corresponding 3,000-gene or 224-ADT feature space, the number of those markers among the top-20 current-checkpoint attribution features, the donor-level M-F1, held-out support and mean fusion weight. The marker lists are representative family tokens, not a complete ontology; availability is not evidence of causal biology.

\scriptsize
\begin{longtable}{@{}p{2.2cm}p{1.0cm}p{0.9cm}rrrrr@{}}
\caption{58-label representative panel-evidence audit. Empty representative marker lists are shown as unavailable rather than 0/0 available.}\label{tab:marker-mechanism}\\
\toprule
Label & Group & Avail. & RNA hits & ADT hits & M-F1 & Support & Fusion $w$ \\
\midrule
\endfirsthead
\toprule
Label & Group & Avail. & RNA hits & ADT hits & M-F1 & Support & Fusion $w$ \\
\midrule
\endhead
""" + "\n".join(rows) + r"""
\bottomrule
\end{longtable}

\noindent\textbf{U1. Five-target panel/rejection diagnostic (unnumbered).}
\begin{center}
\begin{tabular}{@{}lrrrr@{}}
\toprule
Target & Panel representative marker & M-F1 & Unknown recall & AUROC \\
\midrule
""" + "\n".join(target_rows) + r"""
\bottomrule
\end{tabular}
\end{center}
\noindent The five-target table is a descriptive stress test rather than a powered association test. B naive lambda is the only target without a direct kappa/lambda-like ADT marker and also has the lowest unknown recall; RNA light-chain features are present but were not retained among the representative top-20 attribution hits. The target count is too small to infer a general marker--rejection law.

\noindent\textbf{U2. CD8 boundary linkage (unnumbered).}
\begin{center}
\begin{tabular}{@{}lrrrrrrr@{}}
\toprule
PBMC label & PDC101 class & ADT avail./rep. & ADT hits & Fusion $w$ & MOSAIC F1 & MMoCHi F1 & MMoCHi--MOSAIC F1 \\
\midrule
""" + "\n".join(cd8_rows) + r"""
\bottomrule
\end{tabular}
\end{center}
\noindent The CD8 rows connect current PBMC attribution and fusion summaries to the separately evaluated PDC101 hierarchy. MMoCHi's larger F1 in the matched CD8 memory rows is compatible with its curated hierarchy and marker regime, but the label spaces and preprocessing are not identical; this is a boundary comparison, not a causal mediation analysis.
"""
    (table_dir / "supplement_marker_mechanism_audit.tex").write_text(content, encoding="utf-8")


def build_audit(
    donors: list[str],
    seeds: list[int],
    max_per_class: int,
    batch_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sums: dict[tuple[str, str], np.ndarray] = {}
    counts: dict[str, int] = defaultdict(int)
    run_counts: dict[str, int] = defaultdict(int)
    feature_names: dict[str, list[str]] = {}
    panel_names: dict[str, set[str]] = {}
    n_total_runs = 0
    for donor in donors:
        for seed in seeds:
            checkpoint = (
                ROOT
                / "results/exp_generalization/mosaic_n_v33/donor_matrix"
                / f"test_{donor}"
                / f"mosaic_full_seed{seed}"
                / "model.pt"
            )
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            model, arrays, class_names = load_checkpoint(checkpoint)
            model = model.to(device)
            model.eval()
            feature_names["RNA"] = [str(x) for x in arrays["gene_names"]]
            feature_names["ADT"] = [str(x) for x in arrays["protein_names"]]
            panel_names["RNA"] = {normalize_feature(x) for x in feature_names["RNA"]}
            panel_names["ADT"] = {normalize_feature(x) for x in feature_names["ADT"]}
            selected = select_correct_indices(
                model,
                arrays,
                class_names,
                max_per_class=max_per_class,
                seed=8100 + int(donor[1:]) * 100 + seed,
                device=device,
            )
            for class_index, class_label in enumerate(class_names):
                label = str(class_label)
                indices = selected[label]
                if len(indices) == 0:
                    continue
                run_counts[label] += 1
                counts[label] += int(len(indices))
                for start in range(0, len(indices), batch_size):
                    batch = indices[start : start + batch_size]
                    gene = torch.as_tensor(arrays["gene"][batch], dtype=torch.float32, device=device)
                    protein = torch.as_tensor(arrays["protein"][batch], dtype=torch.float32, device=device)
                    gene_attr, protein_attr = attribution_batch(model, gene, protein, class_index)
                    for modality, values in (("RNA", gene_attr), ("ADT", protein_attr)):
                        key = (label, modality)
                        if key not in sums:
                            sums[key] = np.zeros(values.shape[1], dtype=np.float64)
                        sums[key] += values.sum(axis=0)
            n_total_runs += 1
            del model, arrays
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"marker audit donor={donor} seed={seed}", flush=True)

    f1, weights = load_f1_and_weights()
    rows = []
    labels = sorted(f1["class_label"].astype(str).unique())
    for label in labels:
        f1_row = f1[f1["class_label"].eq(label)].iloc[0]
        weight_row = weights[weights["label"].eq(label)]
        for modality in ("RNA", "ADT"):
            key = (label, modality)
            markers = representative_markers(label, modality)
            marker_norm = {normalize_feature(x) for x in markers}
            available = [x for x in markers if normalize_feature(x) in panel_names[modality]]
            if key in sums and counts[label] > 0:
                mean_abs = sums[key] / counts[label]
                names = np.asarray(feature_names[modality], dtype=str)
                order = np.argsort(-mean_abs)[:20]
                top = names[order].tolist()
            else:
                top = []
            top_norm = {normalize_feature(x) for x in top}
            rows.append(
                {
                    "class_label": label,
                    "modality": modality,
                    "marker_group": label_group(label) or "none",
                    "representative_markers": ";".join(markers),
                    "available_panel_markers": ";".join(available),
                    "n_representative_markers": len(markers),
                    "n_available_panel_markers": len(available),
                    "top20_attribution_features": ";".join(top),
                    "top20_marker_hits": len(top_norm & marker_norm),
                    "top20_marker_fraction": (len(top_norm & marker_norm) / len(marker_norm) if marker_norm else np.nan),
                    "n_correct_attribution_cells": counts[label],
                    "n_checkpoint_runs_with_correct_cells": run_counts[label],
                    "n_checkpoint_runs": n_total_runs,
                    "mean_f1": float(f1_row["mean_f1"]),
                    "total_test_support": int(f1_row["total_test_support"]),
                    "mean_rna_weight": float(weight_row["mean_rna_weight"].mean()),
                    "mean_adt_weight": float(weight_row["mean_adt_weight"].mean()),
                    "mean_fusion_weight": float(weight_row["mean_fusion_weight"].mean()),
                    "interpretation": "representative panel evidence; not a complete marker definition or causal attribution",
                    "source": "current MOSAIC full checkpoints, 8 donors x 3 seeds",
                }
            )
    summary = pd.DataFrame(rows)

    unknown = pd.read_csv(ROOT / "results/tables/mosaic_n_v33_unknown_reject_metrics_2026-07-23.csv")
    target_rows = unknown[
        unknown["score"].eq("one_minus_margin")
        & unknown["known_coverage_target"].eq(0.8)
        & unknown["target_label"].isin(["B naive lambda", "CD4 TCM_1", "CD8 TEM_4", "NK_3", "gdT_2"])
    ].copy()
    target_rows = target_rows.groupby("target_label", as_index=False).agg(
        unknown_recall=("unknown_recall", "mean"),
        unknown_auroc=("unknown_auroc", "mean"),
        panel_marker_available=("unknown_recall", "size"),
    )
    availability = pd.read_csv(ROOT / "results/tables/mosaic_n_v7_unknown_target_panel_marker_availability_2026-08-12.csv")
    availability = availability.rename(columns={"target": "target_label"})[["target_label", "available_in_224_adt"]]
    target_rows = target_rows.drop(columns=["panel_marker_available"]).merge(availability, on="target_label", how="left")
    target_rows["panel_marker_available"] = target_rows["available_in_224_adt"].eq("yes")
    target_rows["mean_f1"] = target_rows["target_label"].map(
        f1.set_index("class_label")["mean_f1"].to_dict()
    )
    target_rows["interpretation"] = "five-target descriptive stress-test association; not a powered inferential test"
    return summary, target_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--donors", nargs="+", default=DONORS, choices=DONORS)
    parser.add_argument("--seeds", nargs="+", default=SEEDS, type=int)
    parser.add_argument("--max-per-class", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    if args.require_complete and (set(args.donors) != set(DONORS) or set(args.seeds) != set(SEEDS)):
        raise ValueError("--require-complete requires all eight donors and seeds 41/42/43")
    summary, targets = build_audit(args.donors, args.seeds, args.max_per_class, args.batch_size)
    if args.require_complete:
        if summary["class_label"].nunique() != 58 or len(summary) != 116:
            raise ValueError("current marker audit is not complete for all 58 labels and two modalities")
    out_dir = ROOT / "results/tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"mosaic_n_v8_marker_mechanism_audit_{DATE}.csv"
    target_path = out_dir / f"mosaic_n_v8_marker_target_association_{DATE}.csv"
    summary.to_csv(summary_path, index=False)
    targets.to_csv(target_path, index=False)
    for mirror in (ROOT / "output/tables",):
        mirror.mkdir(parents=True, exist_ok=True)
        summary.to_csv(mirror / summary_path.name, index=False)
        targets.to_csv(mirror / target_path.name, index=False)
    config = {
        "date": DATE,
        "protocol": "current MOSAIC full checkpoints; eight held-out donors x three seeds",
        "attribution": "gradient-times-input on correctly predicted held-out cells",
        "max_correct_cells_per_class_per_run": args.max_per_class,
        "marker_map": "representative panel tokens; no complete marker ontology",
        "forbidden_interpretations": ["causal biomarker", "complete class definition", "universal rejection mechanism"],
        "summary_artifact": str(summary_path.relative_to(ROOT)),
        "target_artifact": str(target_path.relative_to(ROOT)),
    }
    (out_dir / f"mosaic_n_v8_marker_mechanism_audit_{DATE}.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    write_manuscript_fragments(summary, targets)
    print(summary.groupby("modality")["class_label"].nunique().to_string())
    print(targets.to_string(index=False))


if __name__ == "__main__":
    main()
