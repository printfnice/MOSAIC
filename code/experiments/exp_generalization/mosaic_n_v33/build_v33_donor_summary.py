#!/usr/bin/env python
"""Build donor-level V33 summaries without treating seeds as biological replicates."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import balanced_accuracy_score


ROOT = Path(__file__).resolve().parents[3]
DATE = "2026-07-23"
DONORS = [f"P{index}" for index in range(1, 9)]
SEEDS = [41, 42, 43]
RETRAINED_METHODS = ["mlp", "mosaic_full", "mosaic_no_hsr", "mosaic_no_kd"]
METRICS = ["accuracy", "weighted_f1", "macro_f1", "balanced_accuracy"]


def average_seeds_by_donor(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"test_donor", "seed", "method", *METRICS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"run-level frame missing columns: {sorted(missing)}")
    grouped = (
        frame.groupby(["test_donor", "method"], as_index=False)
        .agg(
            **{metric: (metric, "mean") for metric in METRICS},
            n_seeds=("seed", "nunique"),
        )
        .sort_values(["method", "test_donor"])
        .reset_index(drop=True)
    )
    return grouped


def _t_interval(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, mean, 0.0
    margin = float(
        stats.t.ppf(0.975, len(values) - 1)
        * values.std(ddof=1)
        / np.sqrt(len(values))
    )
    return mean - margin, mean + margin, margin


def build_method_summary(donor_means: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, method_frame in donor_means.groupby("method", sort=False):
        for metric in METRICS:
            values = method_frame[metric].to_numpy(dtype=float)
            low, high, margin = _t_interval(values)
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "n_donors": int(len(values)),
                    "mean": float(values.mean()),
                    "sd_across_donors": float(values.std(ddof=1))
                    if len(values) > 1
                    else 0.0,
                    "ci95_low": low,
                    "ci95_high": high,
                    "ci95_margin": margin,
                    "worst_donor_value": float(values.min()),
                    "best_donor_value": float(values.max()),
                }
            )
    return pd.DataFrame(rows)


def average_per_class_seeds_by_donor(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "test_donor",
        "seed",
        "method",
        "class_label",
        "f1",
        "support",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"per-class frame missing columns: {sorted(missing)}")
    observed = frame[frame["support"].astype(float) > 0].copy()
    return (
        observed.groupby(
            ["test_donor", "method", "class_label"],
            as_index=False,
        )
        .agg(
            f1=("f1", "mean"),
            support=("support", "max"),
            n_seeds=("seed", "nunique"),
        )
        .sort_values(["method", "class_label", "test_donor"])
        .reset_index(drop=True)
    )


def build_per_class_summary(donor_class_means: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, class_label), group in donor_class_means.groupby(
        ["method", "class_label"],
        sort=False,
    ):
        values = group["f1"].to_numpy(dtype=float)
        low, high, margin = _t_interval(values)
        worst_index = int(np.argmin(values))
        rows.append(
            {
                "method": method,
                "class_label": class_label,
                "n_observed_donors": int(len(values)),
                "mean_f1": float(values.mean()),
                "sd_across_donors": float(values.std(ddof=1))
                if len(values) > 1
                else 0.0,
                "ci95_low": low,
                "ci95_high": high,
                "ci95_margin": margin,
                "worst_donor": str(group.iloc[worst_index]["test_donor"]),
                "worst_donor_f1": float(values[worst_index]),
                "total_test_support": int(group["support"].sum()),
            }
        )
    return pd.DataFrame(rows)


def _wilcoxon_two_sided(reference: np.ndarray, comparator: np.ndarray) -> float:
    delta = reference - comparator
    if np.allclose(delta, 0.0):
        return 1.0
    return float(stats.wilcoxon(reference, comparator, alternative="two-sided").pvalue)


def build_paired_donor_statistics(
    donor_means: pd.DataFrame,
    reference_method: str,
) -> pd.DataFrame:
    methods = [
        method
        for method in donor_means["method"].astype(str).unique()
        if method != reference_method
    ]
    rows = []
    for comparator in methods:
        pair = donor_means[
            donor_means["method"].isin([reference_method, comparator])
        ]
        for metric in METRICS:
            pivot = pair.pivot(
                index="test_donor",
                columns="method",
                values=metric,
            ).dropna()
            if reference_method not in pivot or comparator not in pivot:
                continue
            reference = pivot[reference_method].to_numpy(dtype=float)
            baseline = pivot[comparator].to_numpy(dtype=float)
            difference = reference - baseline
            low, high, margin = _t_interval(difference)
            if len(difference) > 1 and not np.allclose(difference, difference[0]):
                paired_t_pvalue = float(stats.ttest_rel(reference, baseline).pvalue)
            elif np.allclose(difference, 0.0):
                paired_t_pvalue = 1.0
            else:
                paired_t_pvalue = 0.0
            rows.append(
                {
                    "reference": reference_method,
                    "comparator": comparator,
                    "metric": metric,
                    "n_donors": int(len(difference)),
                    "mean_difference": float(difference.mean()),
                    "difference_ci95_low": low,
                    "difference_ci95_high": high,
                    "difference_ci95_margin": margin,
                    "paired_t_pvalue": paired_t_pvalue,
                    "wilcoxon_pvalue": _wilcoxon_two_sided(
                        reference,
                        baseline,
                    ),
                    "positive_donor_count": int((difference > 0).sum()),
                    "negative_donor_count": int((difference < 0).sum()),
                    "zero_donor_count": int(np.isclose(difference, 0.0).sum()),
                }
            )
    return pd.DataFrame(rows)


def _balanced_accuracy(prediction_path: Path) -> float:
    frame = pd.read_csv(prediction_path, usecols=["label", "prediction"])
    return float(balanced_accuracy_score(frame["label"], frame["prediction"]))


def load_retrained_metrics(
    base_dir: Path,
    donors: list[str],
    seeds: list[int],
    require_complete: bool,
) -> pd.DataFrame:
    rows = []
    missing = []
    for donor in donors:
        for seed in seeds:
            for method in RETRAINED_METHODS:
                run_dir = base_dir / f"test_{donor}" / f"{method}_seed{seed}"
                summary_path = run_dir / "results_summary.csv"
                if not summary_path.exists():
                    missing.append(str(summary_path))
                    continue
                summary = pd.read_csv(summary_path)
                if "eval_mode" in summary.columns:
                    summary = summary[summary["eval_mode"].astype(str).eq("full")]
                if len(summary) != 1:
                    raise ValueError(f"expected one full row in {summary_path}")
                row = summary.iloc[0]
                prediction_name = (
                    "predictions.csv" if method == "mlp" else "predictions_full.csv"
                )
                rows.append(
                    {
                        "test_donor": donor,
                        "seed": seed,
                        "method": method,
                        "accuracy": float(row["test_accuracy"]),
                        "weighted_f1": float(row["test_weighted_f1"]),
                        "macro_f1": float(row["test_macro_f1"]),
                        "balanced_accuracy": _balanced_accuracy(
                            run_dir / prediction_name
                        ),
                        "result_dir": str(run_dir.relative_to(ROOT)),
                    }
                )
    if require_complete and missing:
        raise FileNotFoundError(f"missing {len(missing)} runs: {missing[:3]}")
    return pd.DataFrame(rows)


def load_inference_ablation_metrics(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame.rename(columns={"variant": "method"})
    frame["method"] = "inference::" + frame["method"].astype(str)
    frame["result_dir"] = str(path.parent.relative_to(ROOT))
    return frame[
        [
            "test_donor",
            "seed",
            "method",
            *METRICS,
            "result_dir",
        ]
    ]


def load_retrained_per_class(
    base_dir: Path,
    donors: list[str],
    seeds: list[int],
    require_complete: bool,
) -> pd.DataFrame:
    frames = []
    missing = []
    for donor in donors:
        for seed in seeds:
            for method in RETRAINED_METHODS:
                path = (
                    base_dir
                    / f"test_{donor}"
                    / f"{method}_seed{seed}"
                    / "per_class_metrics.csv"
                )
                if not path.exists():
                    missing.append(str(path))
                    continue
                frame = pd.read_csv(path)
                frame["test_donor"] = donor
                frame["seed"] = seed
                frame["method"] = method
                frames.append(
                    frame[
                        [
                            "test_donor",
                            "seed",
                            "method",
                            "class_label",
                            "f1",
                            "support",
                        ]
                    ]
                )
    if require_complete and missing:
        raise FileNotFoundError(
            f"missing {len(missing)} per-class runs: {missing[:3]}"
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_inference_ablation_per_class(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame.rename(columns={"variant": "method"})
    frame["method"] = "inference::" + frame["method"].astype(str)
    return frame[
        [
            "test_donor",
            "seed",
            "method",
            "class_label",
            "f1",
            "support",
        ]
    ]


def write_package(
    run_level: pd.DataFrame,
    out_dir: Path,
    per_class_run_level: pd.DataFrame | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    donor_means = average_seeds_by_donor(run_level)
    method_summary = build_method_summary(donor_means)
    retrained = donor_means[~donor_means["method"].str.startswith("inference::")]
    inference = donor_means[donor_means["method"].str.startswith("inference::")]
    paired_frames = []
    if "mosaic_full" in set(retrained["method"]):
        paired_frames.append(
            build_paired_donor_statistics(
                retrained,
                reference_method="mosaic_full",
            )
        )
    inference_reference = "inference::margin_gate_hsr"
    if inference_reference in set(inference["method"]):
        paired_frames.append(
            build_paired_donor_statistics(
                inference,
                reference_method=inference_reference,
            )
        )
    paired = (
        pd.concat(paired_frames, ignore_index=True)
        if paired_frames
        else pd.DataFrame()
    )
    run_level.to_csv(out_dir / "run_level_metrics.csv", index=False)
    donor_means.to_csv(out_dir / "donor_seed_averaged_metrics.csv", index=False)
    method_summary.to_csv(out_dir / "method_summary.csv", index=False)
    paired.to_csv(out_dir / "paired_donor_statistics.csv", index=False)
    if per_class_run_level is not None and not per_class_run_level.empty:
        donor_class_means = average_per_class_seeds_by_donor(per_class_run_level)
        per_class_summary = build_per_class_summary(donor_class_means)
        per_class_run_level.to_csv(
            out_dir / "per_class_run_level_metrics.csv",
            index=False,
        )
        donor_class_means.to_csv(
            out_dir / "per_class_donor_seed_averaged_metrics.csv",
            index=False,
        )
        per_class_summary.to_csv(
            out_dir / "per_class_method_summary.csv",
            index=False,
        )
        for base in (ROOT / "results/tables", ROOT / "output/tables"):
            base.mkdir(parents=True, exist_ok=True)
            per_class_summary.to_csv(
                base / f"mosaic_n_v33_donor_per_class_summary_{DATE}.csv",
                index=False,
            )
    for name, frame in {
        f"mosaic_n_v33_donor_method_summary_{DATE}.csv": method_summary,
        f"mosaic_n_v33_paired_donor_statistics_{DATE}.csv": paired,
    }.items():
        for base in (ROOT / "results/tables", ROOT / "output/tables"):
            base.mkdir(parents=True, exist_ok=True)
            frame.to_csv(base / name, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("results/exp_generalization/mosaic_n_v33/donor_matrix"),
    )
    parser.add_argument(
        "--ablation-path",
        type=Path,
        default=Path(
            "results/exp_generalization/mosaic_n_v33/checkpoint_ablations/results_summary.csv"
        ),
    )
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    base_dir = args.base_dir if args.base_dir.is_absolute() else ROOT / args.base_dir
    ablation_path = (
        args.ablation_path
        if args.ablation_path.is_absolute()
        else ROOT / args.ablation_path
    )
    run_level = load_retrained_metrics(
        base_dir,
        DONORS,
        SEEDS,
        args.require_complete,
    )
    if ablation_path.exists():
        run_level = pd.concat(
            [run_level, load_inference_ablation_metrics(ablation_path)],
            ignore_index=True,
        )
    per_class_run_level = load_retrained_per_class(
        base_dir,
        DONORS,
        SEEDS,
        args.require_complete,
    )
    ablation_per_class_path = ablation_path.parent / "per_class_metrics.csv"
    if ablation_per_class_path.exists():
        per_class_run_level = pd.concat(
            [
                per_class_run_level,
                load_inference_ablation_per_class(ablation_per_class_path),
            ],
            ignore_index=True,
        )
    out_dir = ROOT / "results/exp_generalization/mosaic_n_v33/donor_summary"
    write_package(
        run_level,
        out_dir,
        per_class_run_level=per_class_run_level,
    )
    print(build_method_summary(average_seeds_by_donor(run_level)).to_string(index=False))


if __name__ == "__main__":
    main()
