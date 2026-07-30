#!/usr/bin/env python
"""Evaluate validation-selected V33 leave-class-out reject policies."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[3]
LOCAL_DIR = Path(__file__).resolve().parent
GENERALIZATION_V33 = ROOT / "experiments/exp_generalization/mosaic_n_v33"
for directory in (LOCAL_DIR, GENERALIZATION_V33):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from build_v33_unknown_protocol import (  # noqa: E402
    TARGETS,
    safe_target_name,
    target_paths,
    validation_threshold,
)
from evaluate_v33_checkpoint_ablations import load_checkpoint  # noqa: E402


DATE = "2026-07-23"
SEEDS = [41, 42, 43]
COVERAGES = [0.95, 0.80]
SCORES = ["one_minus_max_probability", "one_minus_margin", "energy"]


def evaluation_output_dir(run_dir: Path) -> tuple[Path, bool]:
    formal_dir = ROOT / "results/exp_unknown_celltype/mosaic_n_v33"
    formal_run = run_dir.resolve() == formal_dir.resolve()
    return (
        formal_dir / "evaluation" if formal_run else run_dir / "evaluation",
        formal_run,
    )


def evaluate_score_policy(
    validation_known_scores: np.ndarray,
    test_known_scores: np.ndarray,
    test_unknown_scores: np.ndarray,
    known_coverage: float,
) -> dict[str, float]:
    threshold = validation_threshold(validation_known_scores, known_coverage)
    known_rejected = np.asarray(test_known_scores) > threshold
    unknown_rejected = np.asarray(test_unknown_scores) > threshold
    y_unknown = np.concatenate(
        [
            np.zeros(len(test_known_scores), dtype=int),
            np.ones(len(test_unknown_scores), dtype=int),
        ]
    )
    scores = np.concatenate([test_known_scores, test_unknown_scores])
    return {
        "threshold": float(threshold),
        "known_test_coverage": float(1.0 - known_rejected.mean()),
        "known_false_reject_rate": float(known_rejected.mean()),
        "unknown_recall": float(unknown_rejected.mean()),
        "unknown_auroc": float(roc_auc_score(y_unknown, scores)),
        "unknown_auprc": float(average_precision_score(y_unknown, scores)),
    }


def compute_unknown_scores(
    probabilities: np.ndarray,
    logits: np.ndarray,
) -> dict[str, np.ndarray]:
    sorted_probabilities = np.sort(probabilities, axis=1)
    margin = sorted_probabilities[:, -1] - sorted_probabilities[:, -2]
    maximum = probabilities.max(axis=1)
    logsumexp = np.log(np.exp(logits - logits.max(axis=1, keepdims=True)).sum(axis=1))
    logsumexp = logsumexp + logits.max(axis=1)
    return {
        "one_minus_max_probability": 1.0 - maximum,
        "one_minus_margin": 1.0 - margin,
        "energy": -logsumexp,
    }


def hierarchical_policy_metrics(
    known_l3_correct: np.ndarray,
    known_rejected: np.ndarray,
    known_parent_correct: np.ndarray,
    unknown_rejected: np.ndarray,
    unknown_parent_correct: np.ndarray,
) -> dict[str, float]:
    known_l3_correct = np.asarray(known_l3_correct, dtype=bool)
    known_rejected = np.asarray(known_rejected, dtype=bool)
    known_parent_correct = np.asarray(known_parent_correct, dtype=bool)
    unknown_rejected = np.asarray(unknown_rejected, dtype=bool)
    unknown_parent_correct = np.asarray(unknown_parent_correct, dtype=bool)
    if not (
        len(known_l3_correct)
        == len(known_rejected)
        == len(known_parent_correct)
    ):
        raise ValueError("known hierarchical arrays must have equal length")
    if len(unknown_rejected) != len(unknown_parent_correct):
        raise ValueError("unknown hierarchical arrays must have equal length")
    known_success = (
        (~known_rejected & known_l3_correct)
        | (known_rejected & known_parent_correct)
    )
    unknown_success = unknown_rejected & unknown_parent_correct
    combined = np.concatenate([known_success, unknown_success])
    return {
        "known_hierarchical_accuracy": float(known_success.mean()),
        "unknown_hierarchical_accuracy": float(unknown_success.mean()),
        "combined_hierarchical_accuracy": float(combined.mean()),
        "unknown_unsafe_accept_rate": float((~unknown_rejected).mean()),
    }


@torch.no_grad()
def predict_indices(
    model: torch.nn.Module,
    arrays: dict,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loader = DataLoader(
        TensorDataset(
            torch.as_tensor(arrays["gene"][indices], dtype=torch.float32),
            torch.as_tensor(arrays["protein"][indices], dtype=torch.float32),
            torch.as_tensor(indices, dtype=torch.long),
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    logits = []
    local_indices = []
    for gene, protein, index in loader:
        outputs = model(
            gene.to(device),
            protein.to(device),
            availability_mask=torch.ones(len(index), 2, device=device),
        )
        logits.append(outputs["final_logits"].cpu().numpy())
        local_indices.append(index.numpy())
    logits_array = np.concatenate(logits)
    probabilities = torch.softmax(torch.from_numpy(logits_array), dim=1).numpy()
    return probabilities, logits_array, np.concatenate(local_indices)


def _parent_map() -> dict[str, str]:
    path = (
        ROOT
        / "configs/datasets/pbmc_cite_seq/strict_l3_to_l2_l1_map_seed42.csv"
    )
    frame = pd.read_csv(path)
    return dict(zip(frame["l3_label"].astype(str), frame["l2_label"].astype(str)))


def evaluate_target_seed(
    target: str,
    seed: int,
    checkpoint_path: Path,
    unknown_cache_path: Path,
    batch_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model, arrays, class_names = load_checkpoint(checkpoint_path)
    with np.load(unknown_cache_path, allow_pickle=False) as source:
        unknown_idx = source["test_unknown_idx"].astype(np.int64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    val_prob, val_logits, _ = predict_indices(
        model,
        arrays,
        arrays["val_idx"],
        device,
        batch_size,
    )
    known_prob, known_logits, known_indices = predict_indices(
        model,
        arrays,
        arrays["test_idx"],
        device,
        batch_size,
    )
    unknown_prob, unknown_logits, unknown_indices = predict_indices(
        model,
        arrays,
        unknown_idx,
        device,
        batch_size,
    )
    score_sets = {
        "validation": compute_unknown_scores(val_prob, val_logits),
        "known": compute_unknown_scores(known_prob, known_logits),
        "unknown": compute_unknown_scores(unknown_prob, unknown_logits),
    }
    parents = _parent_map()
    target_parent = parents[target]
    known_pred = class_names[known_prob.argmax(axis=1)]
    unknown_pred = class_names[unknown_prob.argmax(axis=1)]
    known_true = class_names[arrays["labels"][known_indices]]
    known_l3_correct = known_pred == known_true
    known_parent_correct = np.asarray(
        [parents.get(pred) == parents.get(true) for pred, true in zip(known_pred, known_true)]
    )
    unknown_parent_correct = np.asarray(
        [parents.get(pred) == target_parent for pred in unknown_pred]
    )
    rows = []
    prediction_frames = []
    for score_name in SCORES:
        for coverage in COVERAGES:
            policy = evaluate_score_policy(
                score_sets["validation"][score_name],
                score_sets["known"][score_name],
                score_sets["unknown"][score_name],
                coverage,
            )
            threshold = policy["threshold"]
            known_rejected = score_sets["known"][score_name] > threshold
            unknown_rejected = score_sets["unknown"][score_name] > threshold
            known_accepted = ~known_rejected
            hierarchy_metrics = hierarchical_policy_metrics(
                known_l3_correct=known_l3_correct,
                known_rejected=known_rejected,
                known_parent_correct=known_parent_correct,
                unknown_rejected=unknown_rejected,
                unknown_parent_correct=unknown_parent_correct,
            )
            rows.append(
                {
                    "target_label": target,
                    "seed": seed,
                    "score": score_name,
                    "known_coverage_target": coverage,
                    **policy,
                    "known_accepted_l3_accuracy": float(
                        np.mean(known_pred[known_accepted] == known_true[known_accepted])
                    )
                    if known_accepted.any()
                    else np.nan,
                    "known_rejected_parent_accuracy": float(
                        known_parent_correct[known_rejected].mean()
                    )
                    if known_rejected.any()
                    else np.nan,
                    "unknown_rejected_parent_accuracy": float(
                        unknown_parent_correct[unknown_rejected].mean()
                    )
                    if unknown_rejected.any()
                    else np.nan,
                    "unknown_parent_safe_rate": float(
                        np.mean(unknown_rejected & unknown_parent_correct)
                    ),
                    "known_accepted_l3_risk": float(
                        1.0
                        - np.mean(
                            known_pred[known_accepted] == known_true[known_accepted]
                        )
                    )
                    if known_accepted.any()
                    else np.nan,
                    **hierarchy_metrics,
                    "n_val_known": int(len(val_prob)),
                    "n_test_known": int(len(known_prob)),
                    "n_test_unknown": int(len(unknown_prob)),
                }
            )
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "target_label": target,
                        "seed": seed,
                        "score": score_name,
                        "known_coverage_target": coverage,
                        "known_unknown": "unknown",
                        "cell_id": arrays["cell_ids"][unknown_indices],
                        "true_label": target,
                        "prediction": unknown_pred,
                        "uncertainty_score": score_sets["unknown"][score_name],
                        "threshold": threshold,
                        "rejected": unknown_rejected,
                        "parent_correct": unknown_parent_correct,
                    }
                )
            )
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "target_label": target,
                        "seed": seed,
                        "score": score_name,
                        "known_coverage_target": coverage,
                        "known_unknown": "known",
                        "cell_id": arrays["cell_ids"][known_indices],
                        "true_label": known_true,
                        "prediction": known_pred,
                        "uncertainty_score": score_sets["known"][score_name],
                        "threshold": threshold,
                        "rejected": known_rejected,
                        "parent_correct": known_parent_correct,
                    }
                )
            )
    return pd.DataFrame(rows), pd.concat(prediction_frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("results/exp_unknown_celltype/mosaic_n_v33"),
    )
    parser.add_argument("--targets", nargs="+", choices=TARGETS, default=TARGETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--n-genes", type=int, default=3000)
    parser.add_argument("--max-cells", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    metric_frames = []
    prediction_frames = []
    missing = []
    for target in args.targets:
        paths = target_paths(target, args.n_genes, args.max_cells)
        for seed in args.seeds:
            checkpoint = (
                run_dir
                / safe_target_name(target)
                / f"mosaic_full_seed{seed}"
                / "model.pt"
            )
            if not checkpoint.exists() or not paths["unknown"].exists():
                missing.append(str(checkpoint))
                continue
            metrics, predictions = evaluate_target_seed(
                target,
                seed,
                checkpoint,
                paths["unknown"],
                args.batch_size,
            )
            metric_frames.append(metrics)
            prediction_frames.append(predictions)
            print(f"Evaluated unknown target={target}, seed={seed}", flush=True)
    if args.require_complete and missing:
        raise FileNotFoundError(f"missing {len(missing)} unknown runs: {missing[:3]}")
    if not metric_frames:
        raise RuntimeError("no V33 unknown checkpoints were available")
    metrics = pd.concat(metric_frames, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    out_dir, formal_run = evaluation_output_dir(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out_dir / "unknown_reject_metrics.csv", index=False)
    predictions.to_csv(
        out_dir / "unknown_predictions.csv.gz",
        index=False,
        compression="gzip",
    )
    summary = (
        metrics.groupby(["score", "known_coverage_target"], as_index=False)
        .agg(
            n_targets=("target_label", "nunique"),
            n_seeds=("seed", "nunique"),
            mean_known_coverage=("known_test_coverage", "mean"),
            mean_unknown_recall=("unknown_recall", "mean"),
            mean_unknown_auroc=("unknown_auroc", "mean"),
            mean_unknown_auprc=("unknown_auprc", "mean"),
            mean_unknown_parent_safe_rate=("unknown_parent_safe_rate", "mean"),
            mean_known_hierarchical_accuracy=(
                "known_hierarchical_accuracy",
                "mean",
            ),
            mean_combined_hierarchical_accuracy=(
                "combined_hierarchical_accuracy",
                "mean",
            ),
            worst_target_seed_unknown_recall=("unknown_recall", "min"),
        )
    )
    summary.to_csv(out_dir / "unknown_reject_summary.csv", index=False)
    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "date": DATE,
                "targets": args.targets,
                "seeds": args.seeds,
                "scores": SCORES,
                "known_coverage_targets": COVERAGES,
                "threshold_source": "known validation only",
                "missing_runs": missing,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if formal_run:
        for base in (ROOT / "results/tables", ROOT / "output/tables"):
            base.mkdir(parents=True, exist_ok=True)
            metrics.to_csv(
                base / f"mosaic_n_v33_unknown_reject_metrics_{DATE}.csv",
                index=False,
            )
            summary.to_csv(
                base / f"mosaic_n_v33_unknown_reject_summary_{DATE}.csv",
                index=False,
            )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
