#!/usr/bin/env python
"""Quantify donor- and seed-stable MOSAIC-N feature attribution."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[3]
GENERALIZATION_V33 = ROOT / "experiments/exp_generalization/mosaic_n_v33"
if str(GENERALIZATION_V33) not in sys.path:
    sys.path.insert(0, str(GENERALIZATION_V33))

from evaluate_v33_checkpoint_ablations import (  # noqa: E402
    DONORS,
    SEEDS,
    load_checkpoint,
)


DATE = "2026-07-23"
FOCUS_CLASSES = [
    "CD4 TEM_3",
    "CD4 TEM_4",
    "CD4 TCM_3",
    "CD8 Naive",
    "CD8 Naive_2",
    "CD8 TCM_1",
]
CANONICAL_ADT = {
    "CD4 TEM_3": ["CD3", "CD4", "CD45RO", "CD95"],
    "CD4 TEM_4": ["CD3", "CD4", "CD45RO", "CD95"],
    "CD4 TCM_3": ["CD3", "CD4", "CD45RO", "CD27", "CD127"],
    "CD8 Naive": ["CD3", "CD8", "CD45RA", "CD27", "CD127"],
    "CD8 Naive_2": ["CD3", "CD8", "CD45RA", "CD27", "CD127"],
    "CD8 TCM_1": ["CD3", "CD8", "CD45RO", "CD27", "CD127"],
}
CANONICAL_RNA = {
    "CD4 TEM_3": ["IL7R", "LTB", "AQP3", "MAL"],
    "CD4 TEM_4": ["IL7R", "LTB", "AQP3", "MAL"],
    "CD4 TCM_3": ["CCR7", "IL7R", "TCF7", "LTB", "LEF1"],
    "CD8 Naive": ["CCR7", "TCF7", "LEF1", "MAL", "LTB"],
    "CD8 Naive_2": ["CCR7", "TCF7", "LEF1", "MAL", "LTB"],
    "CD8 TCM_1": ["CCR7", "TCF7", "LEF1", "LTB", "IL7R"],
}


def _normalize_feature(value: str) -> str:
    text = str(value).upper().replace("_", "-").strip()
    if text.endswith("-1") or text.endswith("-2"):
        text = text[:-2]
    if text == "CD8A":
        text = "CD8"
    return text


def top_k_jaccard(first: list[str], second: list[str]) -> float:
    first_set = set(first)
    second_set = set(second)
    union = first_set | second_set
    if not union:
        return 1.0
    return float(len(first_set & second_set) / len(union))


def pairwise_feature_stability(
    frame: pd.DataFrame,
    top_k: int = 20,
) -> pd.DataFrame:
    rows = []
    group_columns = ["seed", "class_label", "modality"]
    for keys, group in frame.groupby(group_columns, sort=False):
        seed, class_label, modality = keys
        donors = sorted(group["test_donor"].astype(str).unique())
        for donor_a, donor_b in itertools.combinations(donors, 2):
            first = group[group["test_donor"].eq(donor_a)]
            second = group[group["test_donor"].eq(donor_b)]
            merged = first[["feature", "mean_abs_attribution"]].merge(
                second[["feature", "mean_abs_attribution"]],
                on="feature",
                suffixes=("_a", "_b"),
            )
            spearman = (
                float(
                    stats.spearmanr(
                        merged["mean_abs_attribution_a"],
                        merged["mean_abs_attribution_b"],
                    ).statistic
                )
                if len(merged) >= 3
                else np.nan
            )
            top_a = (
                first.nlargest(top_k, "mean_abs_attribution")["feature"]
                .astype(str)
                .tolist()
            )
            top_b = (
                second.nlargest(top_k, "mean_abs_attribution")["feature"]
                .astype(str)
                .tolist()
            )
            rows.append(
                {
                    "seed": int(seed),
                    "class_label": class_label,
                    "modality": modality,
                    "donor_a": donor_a,
                    "donor_b": donor_b,
                    "n_shared_features": int(len(merged)),
                    "spearman": spearman,
                    "top_k": int(top_k),
                    "top_k_jaccard": top_k_jaccard(top_a, top_b),
                }
            )
    return pd.DataFrame(rows)


def pairwise_seed_stability(
    frame: pd.DataFrame,
    top_k: int = 20,
) -> pd.DataFrame:
    rows = []
    group_columns = ["test_donor", "class_label", "modality"]
    for keys, group in frame.groupby(group_columns, sort=False):
        test_donor, class_label, modality = keys
        seeds = sorted(group["seed"].astype(int).unique())
        for seed_a, seed_b in itertools.combinations(seeds, 2):
            first = group[group["seed"].eq(seed_a)]
            second = group[group["seed"].eq(seed_b)]
            merged = first[["feature", "mean_abs_attribution"]].merge(
                second[["feature", "mean_abs_attribution"]],
                on="feature",
                suffixes=("_a", "_b"),
            )
            spearman = (
                float(
                    stats.spearmanr(
                        merged["mean_abs_attribution_a"],
                        merged["mean_abs_attribution_b"],
                    ).statistic
                )
                if len(merged) >= 3
                else np.nan
            )
            top_a = (
                first.nlargest(top_k, "mean_abs_attribution")["feature"]
                .astype(str)
                .tolist()
            )
            top_b = (
                second.nlargest(top_k, "mean_abs_attribution")["feature"]
                .astype(str)
                .tolist()
            )
            rows.append(
                {
                    "test_donor": str(test_donor),
                    "class_label": class_label,
                    "modality": modality,
                    "seed_a": int(seed_a),
                    "seed_b": int(seed_b),
                    "n_shared_features": int(len(merged)),
                    "spearman": spearman,
                    "top_k": int(top_k),
                    "top_k_jaccard": top_k_jaccard(top_a, top_b),
                }
            )
    return pd.DataFrame(rows)


def canonical_marker_enrichment(
    top_features: list[str],
    available_features: list[str],
    canonical_markers: list[str],
    n_permutations: int,
    seed: int,
) -> dict[str, float | int]:
    available = sorted({_normalize_feature(value) for value in available_features})
    top = {_normalize_feature(value) for value in top_features}
    canonical = {
        _normalize_feature(value)
        for value in canonical_markers
        if _normalize_feature(value) in available
    }
    observed = len(top & canonical)
    if not canonical or not available:
        return {
            "observed_hits": observed,
            "available_canonical_markers": len(canonical),
            "permutation_pvalue": 1.0,
        }
    generator = np.random.default_rng(seed)
    random_hits = []
    sample_size = min(len(canonical), len(available))
    available_array = np.asarray(available, dtype=str)
    for _ in range(n_permutations):
        sampled = set(
            generator.choice(
                available_array,
                size=sample_size,
                replace=False,
            ).tolist()
        )
        random_hits.append(len(top & sampled))
    pvalue = (1 + int(np.sum(np.asarray(random_hits) >= observed))) / (
        n_permutations + 1
    )
    return {
        "observed_hits": int(observed),
        "available_canonical_markers": int(len(canonical)),
        "permutation_pvalue": float(pvalue),
    }


@torch.no_grad()
def select_correct_indices(
    model: torch.nn.Module,
    arrays: dict,
    class_names: np.ndarray,
    focus_classes: list[str],
    device: torch.device,
    max_per_class: int,
    seed: int,
    batch_size: int,
) -> dict[str, np.ndarray]:
    indices = arrays["test_idx"]
    loader = DataLoader(
        TensorDataset(
            torch.as_tensor(arrays["gene"][indices], dtype=torch.float32),
            torch.as_tensor(arrays["protein"][indices], dtype=torch.float32),
            torch.as_tensor(arrays["labels"][indices], dtype=torch.long),
            torch.as_tensor(indices, dtype=torch.long),
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    selected = {label: [] for label in focus_classes}
    for gene, protein, labels, local_indices in loader:
        outputs = model(
            gene.to(device),
            protein.to(device),
            availability_mask=torch.ones(len(labels), 2, device=device),
        )
        predictions = outputs["final_logits"].argmax(dim=1).cpu().numpy()
        true_values = labels.numpy()
        for label in focus_classes:
            class_index = int(np.flatnonzero(class_names == label)[0])
            mask = (true_values == class_index) & (predictions == class_index)
            selected[label].extend(local_indices.numpy()[mask].tolist())
    generator = np.random.default_rng(seed)
    output = {}
    for label, values in selected.items():
        values_array = np.asarray(values, dtype=np.int64)
        if len(values_array) > max_per_class:
            values_array = np.sort(
                generator.choice(
                    values_array,
                    size=max_per_class,
                    replace=False,
                )
            )
        output[label] = values_array
    return output


def gradient_times_input(
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
    gene_gradient, protein_gradient = torch.autograd.grad(
        target,
        (gene, protein),
        retain_graph=False,
        create_graph=False,
    )
    return (
        (gene_gradient * gene).detach().cpu().numpy(),
        (protein_gradient * protein).detach().cpu().numpy(),
    )


def integrated_gradients(
    model: torch.nn.Module,
    gene: torch.Tensor,
    protein: torch.Tensor,
    target_index: int,
    steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    gene_original = gene.detach()
    protein_original = protein.detach()
    gene_gradient_sum = torch.zeros_like(gene_original)
    protein_gradient_sum = torch.zeros_like(protein_original)
    for alpha in torch.linspace(1.0 / steps, 1.0, steps, device=gene.device):
        gene_scaled = (gene_original * alpha).requires_grad_(True)
        protein_scaled = (protein_original * alpha).requires_grad_(True)
        outputs = model(
            gene_scaled,
            protein_scaled,
            availability_mask=torch.ones(len(gene), 2, device=gene.device),
        )
        target = outputs["final_logits"][:, target_index].sum()
        gene_gradient, protein_gradient = torch.autograd.grad(
            target,
            (gene_scaled, protein_scaled),
            retain_graph=False,
            create_graph=False,
        )
        gene_gradient_sum += gene_gradient.detach()
        protein_gradient_sum += protein_gradient.detach()
    return (
        (gene_original * gene_gradient_sum / steps).cpu().numpy(),
        (protein_original * protein_gradient_sum / steps).cpu().numpy(),
    )


def _attribution_rows(
    values: np.ndarray,
    feature_names: list[str],
    donor: str,
    seed: int,
    class_label: str,
    modality: str,
    n_samples: int,
) -> list[dict]:
    mean_abs = np.mean(np.abs(values), axis=0)
    mean_signed = np.mean(values, axis=0)
    return [
        {
            "test_donor": donor,
            "seed": seed,
            "class_label": class_label,
            "modality": modality,
            "feature": str(feature),
            "mean_abs_attribution": float(abs_value),
            "mean_signed_attribution": float(signed_value),
            "n_correct_samples": int(n_samples),
        }
        for feature, abs_value, signed_value in zip(
            feature_names,
            mean_abs,
            mean_signed,
        )
    ]


def analyze_checkpoint(
    checkpoint_path: Path,
    donor: str,
    seed: int,
    max_per_class: int,
    batch_size: int,
    ig_samples: int,
    ig_steps: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model, arrays, class_names = load_checkpoint(checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    selected = select_correct_indices(
        model,
        arrays,
        class_names,
        FOCUS_CLASSES,
        device,
        max_per_class,
        seed=330000 + int(donor[1:]) * 100 + seed,
        batch_size=batch_size,
    )
    rows = []
    ig_rows = []
    for class_label, indices in selected.items():
        if len(indices) == 0:
            continue
        class_index = int(np.flatnonzero(class_names == class_label)[0])
        gene = torch.as_tensor(arrays["gene"][indices], dtype=torch.float32, device=device)
        protein = torch.as_tensor(
            arrays["protein"][indices],
            dtype=torch.float32,
            device=device,
        )
        gene_attr, protein_attr = gradient_times_input(
            model,
            gene,
            protein,
            class_index,
        )
        rows.extend(
            _attribution_rows(
                gene_attr,
                arrays["gene_names"],
                donor,
                seed,
                class_label,
                "RNA",
                len(indices),
            )
        )
        rows.extend(
            _attribution_rows(
                protein_attr,
                arrays["protein_names"],
                donor,
                seed,
                class_label,
                "ADT",
                len(indices),
            )
        )
        if seed == 42 and ig_samples > 0:
            count = min(ig_samples, len(indices))
            ig_gene, ig_protein = integrated_gradients(
                model,
                gene[:count],
                protein[:count],
                class_index,
                steps=ig_steps,
            )
            for modality, gxi, ig in [
                ("RNA", gene_attr[:count], ig_gene),
                ("ADT", protein_attr[:count], ig_protein),
            ]:
                gxi_mean = np.mean(np.abs(gxi), axis=0)
                ig_mean = np.mean(np.abs(ig), axis=0)
                ig_rows.append(
                    {
                        "test_donor": donor,
                        "seed": seed,
                        "class_label": class_label,
                        "modality": modality,
                        "n_samples": count,
                        "steps": ig_steps,
                        "spearman_gxi_vs_ig": float(
                            stats.spearmanr(gxi_mean, ig_mean).statistic
                        ),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(ig_rows)


def build_marker_enrichment(frame: pd.DataFrame, top_k: int = 20) -> pd.DataFrame:
    rows = []
    averaged = (
        frame.groupby(["class_label", "modality", "feature"], as_index=False)[
            "mean_abs_attribution"
        ]
        .mean()
    )
    for (class_label, modality), group in averaged.groupby(
        ["class_label", "modality"]
    ):
        top = (
            group.nlargest(top_k, "mean_abs_attribution")["feature"]
            .astype(str)
            .tolist()
        )
        canonical = (
            CANONICAL_ADT[class_label]
            if modality == "ADT"
            else CANONICAL_RNA[class_label]
        )
        enrichment = canonical_marker_enrichment(
            top_features=top,
            available_features=group["feature"].astype(str).tolist(),
            canonical_markers=canonical,
            n_permutations=5000,
            seed=3300 + FOCUS_CLASSES.index(class_label) * 10 + (modality == "ADT"),
        )
        available_count = int(enrichment["available_canonical_markers"])
        observed_hits = int(enrichment["observed_hits"])
        rows.append(
            {
                "class_label": class_label,
                "modality": modality,
                "top_k": top_k,
                "top_features": ";".join(top),
                "canonical_markers": ";".join(canonical),
                **enrichment,
                "available_canonical_count": available_count,
                "observed_overlap": observed_hits,
                "observed_fraction": (
                    float(observed_hits / available_count)
                    if available_count
                    else 0.0
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("results/exp_generalization/mosaic_n_v33/donor_matrix"),
    )
    parser.add_argument("--donors", nargs="+", choices=DONORS, default=DONORS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--max-per-class", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--ig-samples", type=int, default=8)
    parser.add_argument("--ig-steps", type=int, default=16)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    base_dir = args.base_dir if args.base_dir.is_absolute() else ROOT / args.base_dir
    attribution_frames = []
    ig_frames = []
    missing = []
    for donor in args.donors:
        for seed in args.seeds:
            checkpoint = (
                base_dir / f"test_{donor}" / f"mosaic_full_seed{seed}" / "model.pt"
            )
            if not checkpoint.exists():
                missing.append(str(checkpoint))
                continue
            attribution, ig = analyze_checkpoint(
                checkpoint,
                donor,
                seed,
                args.max_per_class,
                args.batch_size,
                args.ig_samples,
                args.ig_steps,
            )
            attribution_frames.append(attribution)
            ig_frames.append(ig)
            print(f"Attributed donor={donor}, seed={seed}", flush=True)
    if args.require_complete and missing:
        raise FileNotFoundError(f"missing {len(missing)} checkpoints: {missing[:3]}")
    if not attribution_frames:
        raise RuntimeError("no V33 checkpoints were available for attribution")
    attributions = pd.concat(attribution_frames, ignore_index=True)
    ig_sanity = pd.concat(ig_frames, ignore_index=True)
    donor_stability = pairwise_feature_stability(attributions, top_k=20)
    seed_stability = pairwise_seed_stability(attributions, top_k=20)
    marker_enrichment = build_marker_enrichment(attributions, top_k=20)
    out_dir = ROOT / "results/exp_explainability/mosaic_n_v33"
    out_dir.mkdir(parents=True, exist_ok=True)
    attributions.to_csv(
        out_dir / "feature_attributions.csv.gz",
        index=False,
        compression="gzip",
    )
    donor_stability.to_csv(out_dir / "donor_pair_stability.csv", index=False)
    seed_stability.to_csv(out_dir / "seed_pair_stability.csv", index=False)
    marker_enrichment.to_csv(out_dir / "canonical_marker_enrichment.csv", index=False)
    ig_sanity.to_csv(out_dir / "integrated_gradients_sanity.csv", index=False)
    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "date": DATE,
                "focus_classes": FOCUS_CLASSES,
                "max_correct_cells_per_class": args.max_per_class,
                "attribution": "gradient-times-input",
                "integrated_gradients_sanity_samples": args.ig_samples,
                "integrated_gradients_steps": args.ig_steps,
                "marker_enrichment_permutations": 5000,
                "missing_checkpoints": missing,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    for name, frame in {
        f"mosaic_n_v33_donor_attribution_stability_{DATE}.csv": donor_stability,
        f"mosaic_n_v33_marker_enrichment_{DATE}.csv": marker_enrichment,
    }.items():
        for base in (ROOT / "results/tables", ROOT / "output/tables"):
            base.mkdir(parents=True, exist_ok=True)
            frame.to_csv(base / name, index=False)
    print(marker_enrichment.to_string(index=False))


if __name__ == "__main__":
    main()
