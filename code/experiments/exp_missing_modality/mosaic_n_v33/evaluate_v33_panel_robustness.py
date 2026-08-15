#!/usr/bin/env python
"""Evaluate V33 MOSAIC-N checkpoints under locked protein-panel masks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[3]
V33_DIR = ROOT / "experiments/exp_generalization/mosaic_n_v33"
if str(V33_DIR) not in sys.path:
    sys.path.insert(0, str(V33_DIR))

from evaluate_v33_checkpoint_ablations import (  # noqa: E402
    DONORS,
    SEEDS,
    known_label_mask,
    load_checkpoint,
)


DATE = "2026-07-23"
RANDOM_FRACTIONS = [0.10, 0.30, 0.50, 0.70]
MEMORY_MARKERS = ["CD45RA", "CD45RO", "CD27", "CD95", "CD127", "CD28"]
T_CELL_MARKERS = ["CD3", "CD4", "CD8", "CD8A", "TCR", "CD45RA", "CD45RO", "CD27"]


def build_random_feature_mask(
    n_features: int,
    fraction: float,
    seed: int,
) -> np.ndarray:
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("mask fraction must be in [0, 1]")
    n_masked = int(round(n_features * fraction))
    generator = np.random.default_rng(seed)
    indices = generator.choice(n_features, size=n_masked, replace=False)
    mask = np.zeros(n_features, dtype=bool)
    mask[indices] = True
    return mask


def _canonical_marker_name(name: str) -> str:
    return str(name).upper().replace("_", "-").strip()


def marker_feature_mask(
    feature_names: list[str],
    marker_prefixes: list[str],
) -> np.ndarray:
    prefixes = [_canonical_marker_name(value) for value in marker_prefixes]
    def matches(name: str, prefix: str) -> bool:
        canonical = _canonical_marker_name(name)
        return (
            canonical == prefix
            or canonical.startswith(prefix + "-")
            or canonical.startswith(prefix + "/")
        )

    return np.asarray(
        [
            any(matches(name, prefix) for prefix in prefixes)
            for name in feature_names
        ],
        dtype=bool,
    )


def expected_calibration_error(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 15,
) -> float:
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    correct = prediction == y_true
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for index in range(n_bins):
        if index == n_bins - 1:
            mask = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            mask = (confidence >= edges[index]) & (confidence < edges[index + 1])
        if not mask.any():
            continue
        ece += float(mask.mean()) * abs(
            float(correct[mask].mean()) - float(confidence[mask].mean())
        )
    return float(ece)


def multiclass_brier_score(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    targets = np.zeros_like(probabilities)
    targets[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((probabilities - targets) ** 2, axis=1)))


def build_scenarios(
    protein_names: list[str],
    donor: str,
    seed: int,
) -> dict[str, dict]:
    n_features = len(protein_names)
    scenarios = {
        "full": {
            "mask": np.zeros(n_features, dtype=bool),
            "availability": (1.0, 1.0),
            "fraction": 0.0,
        }
    }
    donor_number = int(donor[1:])
    for fraction in RANDOM_FRACTIONS:
        mask_seed = 330000 + donor_number * 1000 + seed * 10 + int(fraction * 10)
        scenarios[f"random_{int(fraction * 100)}"] = {
            "mask": build_random_feature_mask(n_features, fraction, mask_seed),
            "availability": (1.0, 1.0),
            "fraction": fraction,
            "mask_seed": mask_seed,
        }
    scenarios["marker_memory"] = {
        "mask": marker_feature_mask(protein_names, MEMORY_MARKERS),
        "availability": (1.0, 1.0),
        "fraction": np.nan,
    }
    scenarios["marker_tcell"] = {
        "mask": marker_feature_mask(protein_names, T_CELL_MARKERS),
        "availability": (1.0, 1.0),
        "fraction": np.nan,
    }
    scenarios["rna_only"] = {
        "mask": np.ones(n_features, dtype=bool),
        "availability": (1.0, 0.0),
        "fraction": 1.0,
    }
    return scenarios


def _summary(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    prediction = probabilities.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "weighted_f1": float(
            f1_score(y_true, prediction, average="weighted", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_true, prediction, average="macro", zero_division=0)
        ),
        "ece": expected_calibration_error(y_true, probabilities),
        "brier": multiclass_brier_score(y_true, probabilities),
    }


def _per_class(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: np.ndarray,
    donor: str,
    seed: int,
    scenario: str,
) -> pd.DataFrame:
    labels = sorted(np.unique(y_true).tolist())
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    rows = []
    for label_index in labels:
        metrics = report[str(label_index)]
        rows.append(
            {
                "test_donor": donor,
                "seed": seed,
                "scenario": scenario,
                "class_label": str(class_names[label_index]),
                "f1": float(metrics["f1-score"]),
                "recall": float(metrics["recall"]),
                "support": int(metrics["support"]),
            }
        )
    return pd.DataFrame(rows)


@torch.no_grad()
def evaluate_checkpoint_masks(
    checkpoint_path: Path,
    donor: str,
    seed: int,
    batch_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model, arrays, class_names = load_checkpoint(checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    indices = arrays["test_idx"]
    loader = DataLoader(
        TensorDataset(
            torch.as_tensor(arrays["gene"][indices], dtype=torch.float32),
            torch.as_tensor(arrays["protein"][indices], dtype=torch.float32),
            torch.as_tensor(arrays["labels"][indices], dtype=torch.long),
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    scenarios = build_scenarios(arrays["protein_names"], donor, seed)
    metric_rows = []
    class_frames = []
    mask_rows = []
    train_labels = arrays["labels"][arrays["train_idx"]]
    for scenario_name, scenario in scenarios.items():
        probability_chunks = []
        label_chunks = []
        protein_mask = torch.as_tensor(scenario["mask"], dtype=torch.bool, device=device)
        availability_value = torch.tensor(
            scenario["availability"],
            dtype=torch.float32,
            device=device,
        )
        for gene, protein, labels in loader:
            gene = gene.to(device)
            protein = protein.to(device)
            protein[:, protein_mask] = 0.0
            availability = availability_value.unsqueeze(0).expand(len(labels), -1)
            outputs = model(
                gene,
                protein,
                availability_mask=availability,
            )
            probability_chunks.append(
                torch.softmax(outputs["final_logits"], dim=1).cpu().numpy()
            )
            label_chunks.append(labels.numpy())
        probabilities = np.concatenate(probability_chunks)
        y_true = np.concatenate(label_chunks)
        known = known_label_mask(y_true, train_labels)
        probabilities = probabilities[known]
        y_true = y_true[known]
        y_pred = probabilities.argmax(axis=1)
        metric_rows.append(
            {
                "test_donor": donor,
                "seed": seed,
                "scenario": scenario_name,
                "mask_fraction": scenario["fraction"],
                "n_masked_proteins": int(np.sum(scenario["mask"])),
                "n_test": int(len(y_true)),
                **_summary(y_true, probabilities),
            }
        )
        class_frames.append(
            _per_class(
                y_true,
                y_pred,
                class_names,
                donor,
                seed,
                scenario_name,
            )
        )
        for feature_name, masked in zip(arrays["protein_names"], scenario["mask"]):
            if masked:
                mask_rows.append(
                    {
                        "test_donor": donor,
                        "seed": seed,
                        "scenario": scenario_name,
                        "protein": feature_name,
                        "mask_seed": scenario.get("mask_seed", ""),
                    }
                )
    return (
        pd.DataFrame(metric_rows),
        pd.concat(class_frames, ignore_index=True),
        pd.DataFrame(mask_rows),
    )


def add_full_deltas(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for metric in ("accuracy", "weighted_f1", "macro_f1", "ece", "brier"):
        full = output[output["scenario"].eq("full")][
            ["test_donor", "seed", metric]
        ].rename(columns={metric: f"full_{metric}"})
        output = output.merge(full, on=["test_donor", "seed"], how="left")
        output[f"delta_{metric}"] = output[metric] - output[f"full_{metric}"]
    return output


def build_random_mask_slopes(frame: pd.DataFrame) -> pd.DataFrame:
    random_frame = frame[
        frame["scenario"].isin(
            ["full", "random_10", "random_30", "random_50", "random_70"]
        )
    ]
    rows = []
    for (donor, seed), group in random_frame.groupby(["test_donor", "seed"]):
        x = group["mask_fraction"].to_numpy(dtype=float)
        for metric in ("accuracy", "weighted_f1", "macro_f1", "ece", "brier"):
            slope = float(np.polyfit(x, group[metric].to_numpy(dtype=float), 1)[0])
            rows.append(
                {
                    "test_donor": donor,
                    "seed": seed,
                    "metric": metric,
                    "slope_per_full_missing_fraction": slope,
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
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    base_dir = args.base_dir if args.base_dir.is_absolute() else ROOT / args.base_dir
    out_dir = ROOT / "results/exp_missing_modality/mosaic_n_v33"
    out_dir.mkdir(parents=True, exist_ok=True)
    metric_frames = []
    class_frames = []
    mask_frames = []
    missing = []
    for donor in args.donors:
        for seed in args.seeds:
            checkpoint = (
                base_dir / f"test_{donor}" / f"mosaic_full_seed{seed}" / "model.pt"
            )
            if not checkpoint.exists():
                missing.append(str(checkpoint))
                continue
            metrics, per_class, masks = evaluate_checkpoint_masks(
                checkpoint,
                donor,
                seed,
                args.batch_size,
            )
            metric_frames.append(metrics)
            class_frames.append(per_class)
            mask_frames.append(masks)
            print(f"Evaluated panel masks: donor={donor}, seed={seed}", flush=True)
    if args.require_complete and missing:
        raise FileNotFoundError(f"missing {len(missing)} checkpoints: {missing[:3]}")
    if not metric_frames:
        raise RuntimeError("no V33 full checkpoints were available")

    metrics = add_full_deltas(pd.concat(metric_frames, ignore_index=True))
    per_class = pd.concat(class_frames, ignore_index=True)
    masks = pd.concat(mask_frames, ignore_index=True)
    slopes = build_random_mask_slopes(metrics)
    metrics.to_csv(out_dir / "mask_metrics.csv", index=False)
    per_class.to_csv(out_dir / "per_class_metrics.csv", index=False)
    masks.to_csv(out_dir / "mask_manifest.csv", index=False)
    slopes.to_csv(out_dir / "missingness_slopes.csv", index=False)
    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "date": DATE,
                "donors": args.donors,
                "seeds": args.seeds,
                "random_fractions": RANDOM_FRACTIONS,
                "memory_markers": MEMORY_MARKERS,
                "t_cell_markers": T_CELL_MARKERS,
                "mask_policy": "feature-level deterministic masks; no test-label selection",
                "missing_checkpoints": missing,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    for name, frame in {
        f"mosaic_n_v33_panel_robustness_metrics_{DATE}.csv": metrics,
        f"mosaic_n_v33_panel_robustness_slopes_{DATE}.csv": slopes,
    }.items():
        for base in (ROOT / "results/tables", ROOT / "output/tables"):
            base.mkdir(parents=True, exist_ok=True)
            frame.to_csv(base / name, index=False)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
