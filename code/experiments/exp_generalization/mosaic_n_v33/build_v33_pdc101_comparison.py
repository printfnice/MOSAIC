#!/usr/bin/env python
"""Build the caveated same-holdout PDC101 MOSAIC-N/MMoCHi comparison."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score


ROOT = Path(__file__).resolve().parents[3]
DATE = "2026-07-23"
METRICS = ["accuracy", "weighted_f1", "macro_f1"]


def summarize_mosaic_seeds(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in METRICS:
        values = frame[metric].to_numpy(dtype=float)
        n = len(values)
        margin = (
            float(
                stats.t.ppf(0.975, n - 1)
                * values.std(ddof=1)
                / np.sqrt(n)
            )
            if n > 1
            else 0.0
        )
        rows.append(
            {
                "method": "MOSAIC-N",
                "metric": metric,
                "n_seeds": n,
                "mean": float(values.mean()),
                "ci95_margin": margin,
                "uncertainty_mode": "seed Student-t",
            }
        )
    return pd.DataFrame(rows)


def summarize_mosaic_per_class(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for class_label, group in frame.groupby("class_label", sort=True):
        values = group["f1"].to_numpy(dtype=float)
        n = len(values)
        margin = (
            float(
                stats.t.ppf(0.975, n - 1)
                * values.std(ddof=1)
                / np.sqrt(n)
            )
            if n > 1
            else 0.0
        )
        rows.append(
            {
                "method": "MOSAIC-N",
                "class_label": class_label,
                "n_seeds": n,
                "mean_f1": float(values.mean()),
                "ci95_margin": margin,
                "support": int(group["support"].max()),
                "uncertainty_mode": "seed Student-t",
            }
        )
    return pd.DataFrame(rows)


def _bootstrap_mmochi(
    predictions: pd.DataFrame,
    n_bootstrap: int = 2000,
    seed: int = 3301,
) -> pd.DataFrame:
    y_true = predictions["sort_label"].astype(str).to_numpy()
    y_pred = predictions["mmochi_prediction"].astype(str).to_numpy()
    generator = np.random.default_rng(seed)
    estimates = {metric: [] for metric in METRICS}
    for _ in range(n_bootstrap):
        indices = generator.integers(0, len(y_true), size=len(y_true))
        true_sample = y_true[indices]
        pred_sample = y_pred[indices]
        estimates["accuracy"].append(accuracy_score(true_sample, pred_sample))
        estimates["weighted_f1"].append(
            f1_score(true_sample, pred_sample, average="weighted", zero_division=0)
        )
        estimates["macro_f1"].append(
            f1_score(true_sample, pred_sample, average="macro", zero_division=0)
        )
    point = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
    }
    rows = []
    for metric in METRICS:
        values = np.asarray(estimates[metric], dtype=float)
        low, high = np.quantile(values, [0.025, 0.975])
        rows.append(
            {
                "method": "MMoCHi",
                "metric": metric,
                "n_seeds": 1,
                "mean": point[metric],
                "ci95_margin": float(max(point[metric] - low, high - point[metric])),
                "uncertainty_mode": "cell bootstrap; conditional on one workflow",
            }
        )
    return pd.DataFrame(rows)


def _bootstrap_mmochi_per_class(
    predictions: pd.DataFrame,
    n_bootstrap: int = 2000,
    seed: int = 3302,
) -> pd.DataFrame:
    y_true = predictions["sort_label"].astype(str).to_numpy()
    y_pred = predictions["mmochi_prediction"].astype(str).to_numpy()
    labels = np.asarray(sorted(np.unique(y_true)), dtype=str)
    point = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )
    generator = np.random.default_rng(seed)
    bootstrap = np.empty((n_bootstrap, len(labels)), dtype=float)
    for index in range(n_bootstrap):
        sample = generator.integers(0, len(y_true), size=len(y_true))
        bootstrap[index] = f1_score(
            y_true[sample],
            y_pred[sample],
            labels=labels,
            average=None,
            zero_division=0,
        )
    low = np.quantile(bootstrap, 0.025, axis=0)
    high = np.quantile(bootstrap, 0.975, axis=0)
    support = pd.Series(y_true).value_counts()
    return pd.DataFrame(
        {
            "method": "MMoCHi",
            "class_label": labels,
            "n_seeds": 1,
            "mean_f1": point,
            "ci95_margin": np.maximum(point - low, high - point),
            "support": [int(support[label]) for label in labels],
            "uncertainty_mode": "cell bootstrap; conditional on one workflow",
        }
    )


def load_mosaic_seed_metrics(base_dir: Path) -> pd.DataFrame:
    rows = []
    for seed in (41, 42, 43):
        path = base_dir / f"mosaic_full_seed{seed}" / "results_summary.csv"
        frame = pd.read_csv(path)
        frame = frame[frame["eval_mode"].astype(str).eq("full")]
        if len(frame) != 1:
            raise ValueError(f"expected one full MOSAIC-N row in {path}")
        row = frame.iloc[0]
        rows.append(
            {
                "seed": seed,
                "accuracy": float(row["test_accuracy"]),
                "weighted_f1": float(row["test_weighted_f1"]),
                "macro_f1": float(row["test_macro_f1"]),
            }
        )
    return pd.DataFrame(rows)


def load_mosaic_per_class_metrics(base_dir: Path) -> pd.DataFrame:
    frames = []
    for seed in (41, 42, 43):
        path = base_dir / f"mosaic_full_seed{seed}" / "per_class_metrics.csv"
        frame = pd.read_csv(path)
        frame["seed"] = seed
        frames.append(frame[["seed", "class_label", "f1", "support"]])
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    mosaic_dir = ROOT / "results/exp_generalization/mosaic_n_v33/pdc101"
    mmochi_path = (
        ROOT
        / "results/exp_generalization/mmochi_pdc101_sorted_ext_holdout_thresholds/holdout_predictions.csv"
    )
    mosaic_seeds = load_mosaic_seed_metrics(mosaic_dir)
    mosaic_summary = summarize_mosaic_seeds(mosaic_seeds)
    mosaic_per_class = summarize_mosaic_per_class(
        load_mosaic_per_class_metrics(mosaic_dir)
    )
    mmochi_predictions = pd.read_csv(mmochi_path)
    mmochi_summary = _bootstrap_mmochi(mmochi_predictions)
    mmochi_per_class = _bootstrap_mmochi_per_class(mmochi_predictions)
    comparison = pd.concat([mosaic_summary, mmochi_summary], ignore_index=True)
    comparison["dataset"] = "PDC101 sorted external holdout"
    comparison["n_holdout"] = 2098
    comparison["protocol_caveat"] = comparison["method"].map(
        {
            "MOSAIC-N": "train-only preprocessing; no HTO/control; three model seeds",
            "MMoCHi": "official hierarchy; external_holdout=True; one workflow fit",
        }
    )
    per_class_comparison = pd.concat(
        [mosaic_per_class, mmochi_per_class],
        ignore_index=True,
    )
    per_class_comparison["dataset"] = "PDC101 sorted external holdout"
    per_class_comparison["n_holdout"] = 2098
    per_class_comparison["protocol_caveat"] = per_class_comparison["method"].map(
        {
            "MOSAIC-N": "train-only preprocessing; no HTO/control; three model seeds",
            "MMoCHi": "official hierarchy; external_holdout=True; one workflow fit",
        }
    )
    out_dir = ROOT / "results/exp_generalization/mosaic_n_v33/pdc101_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    mosaic_seeds.to_csv(out_dir / "mosaic_seed_metrics.csv", index=False)
    comparison.to_csv(out_dir / "comparison_summary.csv", index=False)
    per_class_comparison.to_csv(
        out_dir / "per_class_comparison.csv",
        index=False,
    )
    for base in (ROOT / "results/tables", ROOT / "output/tables"):
        base.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(
            base / f"mosaic_n_v33_pdc101_mmochi_comparison_{DATE}.csv",
            index=False,
        )
        per_class_comparison.to_csv(
            base / f"mosaic_n_v33_pdc101_mmochi_per_class_{DATE}.csv",
            index=False,
        )
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
