#!/usr/bin/env python
"""Run the V8.2 PBMC frozen-checkpoint size-matched ADT null experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[3]
V33_DIR = ROOT / "experiments/exp_generalization/mosaic_n_v33"
PANEL_DIR = ROOT / "experiments/exp_missing_modality/mosaic_n_v33"
for import_dir in (V33_DIR, PANEL_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from evaluate_v33_checkpoint_ablations import (  # noqa: E402
    DONORS,
    SEEDS,
    known_label_mask,
    load_checkpoint,
)
from evaluate_v33_panel_robustness import (  # noqa: E402
    MEMORY_MARKERS,
    T_CELL_MARKERS,
    build_random_feature_mask,
    build_scenarios as build_existing_scenarios,
    expected_calibration_error,
    marker_feature_mask,
    multiclass_brier_score,
)


DATE = "2026-08-16"
RANDOM_SIZE_SEED_BASE = 820000
METRICS = (
    "accuracy",
    "weighted_f1",
    "macro_f1",
    "balanced_accuracy",
    "ece",
    "brier",
)


def build_random_feature_mask_count(
    n_features: int,
    n_masked: int,
    seed: int,
) -> np.ndarray:
    """Build a feature-name-only mask with an exact number of masked features."""
    if not 0 <= n_masked <= n_features:
        raise ValueError(
            f"requested {n_masked} masked features for {n_features} available features"
        )
    generator = np.random.default_rng(seed)
    indices = generator.choice(n_features, size=n_masked, replace=False)
    mask = np.zeros(n_features, dtype=bool)
    mask[indices] = True
    return mask


def build_scenarios(
    protein_names: Iterable[str],
    donor: str,
    seed: int,
) -> dict[str, dict]:
    """Preserve v33 scenarios and add size-matched random null conditions."""
    feature_names = list(map(str, protein_names))
    scenarios = build_existing_scenarios(feature_names, donor, seed)
    target_counts = {
        "random_size_matched_6": int(scenarios["marker_memory"]["mask"].sum()),
        "random_size_matched_15": int(scenarios["marker_tcell"]["mask"].sum()),
    }
    donor_number = int(str(donor).lstrip("P"))
    for offset, (name, n_masked) in enumerate(target_counts.items(), start=1):
        mask_seed = RANDOM_SIZE_SEED_BASE + donor_number * 1000 + seed * 10 + offset
        mask = build_random_feature_mask_count(
            len(feature_names),
            n_masked,
            mask_seed,
        )
        scenarios[name] = {
            "mask": mask,
            "availability": (1.0, 1.0),
            "fraction": float(n_masked / len(feature_names)),
            "mask_seed": mask_seed,
            "mask_source": "random_feature_name_only_size_matched_to_targeted_mask",
        }
    validate_scenario_masks(scenarios, feature_names)
    return scenarios


def validate_scenario_masks(
    scenarios: dict[str, dict],
    protein_names: Iterable[str],
) -> None:
    """Assert mask shape/provenance invariants before any model inference."""
    n_features = len(list(protein_names))
    for name, scenario in scenarios.items():
        mask = np.asarray(scenario["mask"], dtype=bool)
        if mask.shape != (n_features,):
            raise ValueError(f"{name}: mask shape {mask.shape} != {(n_features,)}")
        if name.startswith("random_size_matched_"):
            mask_seed = scenario.get("mask_seed")
            if not isinstance(mask_seed, (int, np.integer)):
                raise ValueError(f"{name}: mask_seed must be an integer provenance value")
            if scenario.get("mask_source") != (
                "random_feature_name_only_size_matched_to_targeted_mask"
            ):
                raise ValueError(f"{name}: invalid or missing mask source")
        if "label" in scenario or "prediction" in scenario:
            raise ValueError(f"{name}: mask metadata contains outcome-dependent fields")


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
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
        pin_memory=device.type == "cuda",
    )
    scenarios = build_scenarios(arrays["protein_names"], donor, seed)
    metric_rows = []
    class_frames = []
    mask_rows = []
    mask_summary_rows = []
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
            gene = gene.to(device, non_blocking=True)
            protein = protein.to(device, non_blocking=True)
            protein[:, protein_mask] = 0.0
            availability = availability_value.unsqueeze(0).expand(len(labels), -1)
            outputs = model(gene, protein, availability_mask=availability)
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
        mask_seed = scenario.get("mask_seed", "")
        mask_source = scenario.get(
            "mask_source",
            "legacy_v33_deterministic_feature_mask",
        )
        metric_rows.append(
            {
                "test_donor": donor,
                "seed": seed,
                "scenario": scenario_name,
                "mask_fraction": scenario["fraction"],
                "n_masked_proteins": int(np.sum(scenario["mask"])),
                "mask_seed": mask_seed,
                "mask_source": mask_source,
                "n_test": int(len(y_true)),
                **_summary(y_true, probabilities),
            }
        )
        class_frames.append(
            _per_class(y_true, y_pred, class_names, donor, seed, scenario_name)
        )
        mask_summary_rows.append(
            {
                "test_donor": donor,
                "seed": seed,
                "scenario": scenario_name,
                "n_masked_proteins": int(np.sum(scenario["mask"])),
                "mask_seed": mask_seed,
                "mask_source": mask_source,
                "masked_proteins": "|".join(
                    str(name)
                    for name, masked in zip(arrays["protein_names"], scenario["mask"])
                    if masked
                ),
            }
        )
        for feature_name, masked in zip(arrays["protein_names"], scenario["mask"]):
            if masked:
                mask_rows.append(
                    {
                        "test_donor": donor,
                        "seed": seed,
                        "scenario": scenario_name,
                        "protein": str(feature_name),
                        "mask_seed": mask_seed,
                        "mask_source": mask_source,
                        "n_masked_proteins": int(np.sum(scenario["mask"])),
                    }
                )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return (
        pd.DataFrame(metric_rows),
        pd.concat(class_frames, ignore_index=True),
        pd.DataFrame(mask_rows),
        pd.DataFrame(mask_summary_rows),
    )


def add_full_deltas(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for metric in METRICS:
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
        for metric in METRICS:
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


def _write_readme(out_dir: Path, config: dict) -> None:
    (out_dir / "README.md").write_text(
        """# V8.2 PBMC size-matched random ADT null

This artifact evaluates frozen MOSAIC-N v33 checkpoints under legacy v33
panel masks plus random masks with exactly the same feature counts as the
six-feature memory-marker and fifteen-feature T-cell targeted masks.

Masks are generated from the eligible protein feature names and declared
donor/checkpoint seed only. They are fixed across cells within a unit and
never use test labels, predictions, thresholds, or test errors. Metrics are
computed on labels observed in the corresponding training donors, matching
the v33 known-label policy. This is a protocol/interpretability null, not a
predefined performance-improvement claim.

The raw run is reproducible with the command recorded in `config.yaml` and
the unit-level provenance is recorded in `split_seed_metadata.json`.
""",
        encoding="utf-8",
    )


def _write_yaml(path: Path, config: dict) -> None:
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")


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
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "results/experiments/v8.2_missing_modality_pdc_audit/pbmc_random_adt_null"
        ),
    )
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    base_dir = args.base_dir if args.base_dir.is_absolute() else ROOT / args.base_dir
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "date": DATE,
        "command": " ".join(sys.argv),
        "base_dir": str(base_dir),
        "donors": args.donors,
        "seeds": args.seeds,
        "batch_size": args.batch_size,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "legacy_scenarios": [
            "full",
            "random_10",
            "random_30",
            "random_50",
            "random_70",
            "marker_memory",
            "marker_tcell",
            "rna_only",
        ],
        "new_scenarios": ["random_size_matched_6", "random_size_matched_15"],
        "random_size_seed_base": RANDOM_SIZE_SEED_BASE,
        "mask_policy": (
            "feature-name-only fixed masks; size matched to the existing targeted "
            "six-feature and fifteen-feature masks; no test-label selection"
        ),
        "known_label_policy": "metrics use only labels observed in training donors",
    }
    _write_yaml(out_dir / "config.yaml", config)
    checkpoint_paths = [
        base_dir / f"test_{donor}" / f"mosaic_full_seed{seed}" / "model.pt"
        for donor in args.donors
        for seed in args.seeds
    ]
    (out_dir / "preflight.json").write_text(
        json.dumps(
            {
                "status": "ready" if all(path.exists() for path in checkpoint_paths) else "incomplete",
                "gpu_available": bool(torch.cuda.is_available()),
                "torch_version": torch.__version__,
                "checkpoint_count": int(sum(path.exists() for path in checkpoint_paths)),
                "expected_checkpoint_count": len(checkpoint_paths),
                "missing_checkpoints": [str(path) for path in checkpoint_paths if not path.exists()],
                "base_dir": str(base_dir),
                "donors": args.donors,
                "seeds": args.seeds,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "split_seed_metadata.json").write_text(
        json.dumps(
            {
                "split_unit": "PBMC donor-disjoint v33 frozen checkpoint",
                "checkpoint_units": [
                    f"test_{donor}/mosaic_full_seed{seed}/model.pt"
                    for donor in args.donors
                    for seed in args.seeds
                ],
                "donors": args.donors,
                "seeds": args.seeds,
                "mask_seed_formula": "820000 + donor_number*1000 + model_seed*10 + condition_offset",
                "source_evaluator": str(
                    ROOT / "experiments/exp_missing_modality/mosaic_n_v33/evaluate_v33_panel_robustness.py"
                ),
                "checkpoint_source": str(base_dir),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_readme(out_dir, config)

    metric_frames = []
    class_frames = []
    mask_frames = []
    mask_summary_frames = []
    missing = []
    log_lines = []

    for donor in args.donors:
        for seed in args.seeds:
            checkpoint = base_dir / f"test_{donor}" / f"mosaic_full_seed{seed}" / "model.pt"
            if not checkpoint.exists():
                missing.append(str(checkpoint))
                continue
            metrics, per_class, masks, mask_summary = evaluate_checkpoint_masks(
                checkpoint,
                donor,
                seed,
                args.batch_size,
            )
            metric_frames.append(metrics)
            class_frames.append(per_class)
            mask_frames.append(masks)
            mask_summary_frames.append(mask_summary)
            message = f"Evaluated PBMC size-matched masks: donor={donor}, seed={seed}"
            print(message, flush=True)
            log_lines.append(message)

    if args.require_complete and missing:
        (out_dir / "missing_checkpoints.json").write_text(
            json.dumps(missing, indent=2),
            encoding="utf-8",
        )
        raise FileNotFoundError(f"missing {len(missing)} checkpoints: {missing[:3]}")
    if not metric_frames:
        raise RuntimeError("no V33 full checkpoints were available")

    metrics = add_full_deltas(pd.concat(metric_frames, ignore_index=True))
    per_class = pd.concat(class_frames, ignore_index=True)
    masks = pd.concat(mask_frames, ignore_index=True)
    mask_summary = pd.concat(mask_summary_frames, ignore_index=True)
    slopes = build_random_mask_slopes(metrics)
    metrics.to_csv(out_dir / "results_summary.csv", index=False)
    per_class.to_csv(out_dir / "per_class_metrics.csv", index=False)
    masks.to_csv(out_dir / "mask_manifest.csv", index=False)
    mask_summary.to_csv(out_dir / "mask_summary.csv", index=False)
    slopes.to_csv(out_dir / "missingness_slopes.csv", index=False)
    (out_dir / "missing_checkpoints.json").write_text(
        json.dumps(missing, indent=2),
        encoding="utf-8",
    )
    log_lines.append(f"missing_checkpoints={len(missing)}")
    log_lines.append(f"metric_rows={len(metrics)}")
    log_lines.append(f"per_class_rows={len(per_class)}")
    (out_dir / "run.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
