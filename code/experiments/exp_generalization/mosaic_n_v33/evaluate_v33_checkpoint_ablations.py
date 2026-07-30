#!/usr/bin/env python
"""Evaluate inference-time branch, gate and HSR ablations for V33 checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[3]
STRICT_DIR = ROOT / "experiments/exp_generalization/original_mosaic_strict"
for directory in (STRICT_DIR,):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from run_mosaic_rd_v2 import MosaicRDV2Model, parse_hidden_dims  # noqa: E402
from strict_array_cache import load_strict_arrays_cache  # noqa: E402


DATE = "2026-07-23"
DONORS = [f"P{index}" for index in range(1, 9)]
SEEDS = [41, 42, 43]
VARIANTS = [
    "rna_branch",
    "adt_branch",
    "fusion_branch",
    "uniform_fusion",
    "margin_gate",
    "margin_gate_hsr",
]


def select_variant_logits(
    outputs: dict[str, torch.Tensor],
    variant: str,
) -> torch.Tensor:
    if variant == "rna_branch":
        return outputs["rna_logits"]
    if variant == "adt_branch":
        return outputs["adt_logits"]
    if variant == "fusion_branch":
        return outputs["fusion_logits"]
    if variant == "uniform_fusion":
        return (
            outputs["rna_logits"]
            + outputs["adt_logits"]
            + outputs["fusion_logits"]
        ) / 3.0
    if variant == "margin_gate":
        return outputs["base_final_logits"]
    if variant == "margin_gate_hsr":
        return outputs["final_logits"]
    raise ValueError(f"unsupported V33 ablation variant: {variant}")


def known_label_mask(y_true: np.ndarray, train_labels: np.ndarray) -> np.ndarray:
    return np.isin(np.asarray(y_true), np.unique(np.asarray(train_labels)))


def summarize_variant_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def load_checkpoint(
    checkpoint_path: Path,
) -> tuple[MosaicRDV2Model, dict, np.ndarray]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    args = checkpoint["args"]
    cache_path = Path(args["cache_path"])
    if not cache_path.is_absolute():
        cache_path = ROOT / cache_path
    arrays = load_strict_arrays_cache(cache_path)
    class_names = np.asarray(checkpoint["label_classes"], dtype=str)
    model = MosaicRDV2Model(
        gene_dim=arrays["gene"].shape[1],
        protein_dim=arrays["protein"].shape[1],
        hidden_dim=int(args["hidden_dim"]),
        encoder_hidden_dims=parse_hidden_dims(args["encoder_hidden_dims"]),
        fusion_hidden_dims=parse_hidden_dims(args["fusion_hidden_dims"]),
        num_classes=len(class_names),
        dropout=float(args["dropout"]),
        gate_temperature=float(args["gate_temperature"]),
        head_type=str(args["head_type"]),
        class_names=class_names.tolist(),
        hsr_config=checkpoint["train_info"]["hsr_config"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, arrays, class_names


def _per_class_rows(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: np.ndarray,
    donor: str,
    seed: int,
    variant: str,
) -> list[dict]:
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
                "variant": variant,
                "class_label": str(class_names[label_index]),
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1": float(metrics["f1-score"]),
                "support": int(metrics["support"]),
            }
        )
    return rows


@torch.no_grad()
def evaluate_checkpoint(
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
            torch.as_tensor(indices, dtype=torch.long),
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    y_chunks = []
    index_chunks = []
    probability_chunks = {variant: [] for variant in VARIANTS}
    for gene, protein, labels, local_indices in loader:
        outputs = model(
            gene.to(device),
            protein.to(device),
            availability_mask=torch.ones(len(labels), 2, device=device),
        )
        y_chunks.append(labels.numpy())
        index_chunks.append(local_indices.numpy())
        for variant in VARIANTS:
            probability_chunks[variant].append(
                torch.softmax(select_variant_logits(outputs, variant), dim=1)
                .cpu()
                .numpy()
            )

    y_true = np.concatenate(y_chunks)
    local_indices = np.concatenate(index_chunks)
    known = known_label_mask(y_true, arrays["labels"][arrays["train_idx"]])
    metric_rows = []
    per_class_rows = []
    prediction_frames = []
    for variant in VARIANTS:
        probabilities = np.concatenate(probability_chunks[variant])
        y_pred = probabilities.argmax(axis=1)
        summary = summarize_variant_predictions(y_true[known], y_pred[known])
        metric_rows.append(
            {
                "test_donor": donor,
                "seed": seed,
                "variant": variant,
                "n_test": int(len(y_true)),
                "n_known_test": int(known.sum()),
                "n_unknown_test": int((~known).sum()),
                **summary,
            }
        )
        per_class_rows.extend(
            _per_class_rows(
                y_true[known],
                y_pred[known],
                class_names,
                donor,
                seed,
                variant,
            )
        )
        prediction_frames.append(
            pd.DataFrame(
                {
                    "test_donor": donor,
                    "seed": seed,
                    "variant": variant,
                    "cell_id": arrays["cell_ids"][local_indices],
                    "label": class_names[y_true],
                    "prediction": class_names[y_pred],
                    "confidence": probabilities.max(axis=1),
                    "known_to_train": known,
                }
            )
        )
    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(per_class_rows),
        pd.concat(prediction_frames, ignore_index=True),
    )


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
    out_dir = ROOT / "results/exp_generalization/mosaic_n_v33/checkpoint_ablations"
    out_dir.mkdir(parents=True, exist_ok=True)
    metric_frames = []
    class_frames = []
    prediction_frames = []
    missing = []
    for donor in args.donors:
        for seed in args.seeds:
            checkpoint = (
                base_dir / f"test_{donor}" / f"mosaic_full_seed{seed}" / "model.pt"
            )
            if not checkpoint.exists():
                missing.append(str(checkpoint))
                continue
            metrics, per_class, predictions = evaluate_checkpoint(
                checkpoint,
                donor,
                seed,
                args.batch_size,
            )
            metric_frames.append(metrics)
            class_frames.append(per_class)
            prediction_frames.append(predictions)
            print(f"Evaluated checkpoint ablations: donor={donor}, seed={seed}", flush=True)
    if args.require_complete and missing:
        raise FileNotFoundError(f"missing {len(missing)} checkpoints: {missing[:3]}")
    if not metric_frames:
        raise RuntimeError("no V33 full checkpoints were available")

    metrics = pd.concat(metric_frames, ignore_index=True)
    per_class = pd.concat(class_frames, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics.to_csv(out_dir / "results_summary.csv", index=False)
    per_class.to_csv(out_dir / "per_class_metrics.csv", index=False)
    predictions.to_csv(out_dir / "predictions.csv.gz", index=False, compression="gzip")
    config = {
        "date": DATE,
        "base_dir": str(base_dir),
        "donors": args.donors,
        "seeds": args.seeds,
        "variants": VARIANTS,
        "missing_checkpoints": missing,
        "known_label_policy": "metrics use only labels observed in training donors",
    }
    (out_dir / "config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )
    metrics.to_csv(
        ROOT / "results/tables/mosaic_n_v33_checkpoint_ablation_metrics_2026-07-23.csv",
        index=False,
    )
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
