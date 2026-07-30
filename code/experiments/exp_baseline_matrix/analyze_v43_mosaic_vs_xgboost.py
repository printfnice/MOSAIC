#!/usr/bin/env python
"""Compute donor-paired MOSAIC full versus Early-fusion XGBoost statistics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
DATE = "2026-07-29"
DONORS = [f"P{i}" for i in range(1, 9)]
SEEDS = [41, 42, 43]
METRICS = ["accuracy", "weighted_f1", "macro_f1", "balanced_accuracy"]

DEFAULT_MOSAIC_DIR = ROOT / "results/exp_generalization/mosaic_n_v33/donor_matrix"
DEFAULT_XGB_RUN_LEVEL = ROOT / "results/exp_baseline_matrix/v40_pbmc_donor_full_training_xgb_centroid_gnb/run_level_metrics.csv"
DEFAULT_OUT_DIR = ROOT / "results/exp_baseline_matrix/v43_mosaic_vs_xgboost_paired"
TABLE_DIR = ROOT / "manufacture/mosaic_n_bioinformatics_manuscript_v1/oup-authoring-template/tables/v43"


def load_mosaic_donor_metrics(mosaic_dir: Path) -> pd.DataFrame:
    rows = []
    for donor in DONORS:
        for seed in SEEDS:
            path = mosaic_dir / f"test_{donor}" / f"mosaic_full_seed{seed}" / "results_summary.csv"
            if not path.exists():
                raise FileNotFoundError(path)
            frame = pd.read_csv(path)
            full = frame[frame["eval_mode"].eq("full")]
            if len(full) != 1:
                raise ValueError(f"expected one full row in {path}, found {len(full)}")
            row = full.iloc[0]
            rows.append(
                {
                    "method": "MOSAIC full",
                    "test_donor": donor,
                    "seed": seed,
                    "accuracy": float(row["test_accuracy"]),
                    "weighted_f1": float(row["test_weighted_f1"]),
                    "macro_f1": float(row["test_macro_f1"]),
                }
            )
    seed_level = pd.DataFrame(rows)
    donor_level = seed_level.groupby(["method", "test_donor"], as_index=False)[["accuracy", "weighted_f1", "macro_f1"]].mean()
    return donor_level


def load_xgboost_donor_metrics(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    rows = frame[frame["method"].eq("Early-fusion XGBoost")].copy()
    if len(rows) != len(DONORS):
        raise ValueError(f"expected {len(DONORS)} Early-fusion XGBoost rows, found {len(rows)}")
    return rows[["method", "test_donor", "accuracy", "weighted_f1", "macro_f1", "balanced_accuracy"]]


def paired_metric_rows(mosaic: pd.DataFrame, comparator: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    merged = mosaic.merge(comparator, on="test_donor", suffixes=("_mosaic", "_comparator"))
    rows = []
    for metric in metrics:
        diff = merged[f"{metric}_mosaic"].to_numpy(dtype=float) - merged[f"{metric}_comparator"].to_numpy(dtype=float)
        n = len(diff)
        sd = float(diff.std(ddof=1)) if n > 1 else 0.0
        margin = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)) if n > 1 else 0.0
        pvalue = float(stats.ttest_rel(merged[f"{metric}_mosaic"], merged[f"{metric}_comparator"]).pvalue) if n > 1 else np.nan
        try:
            wilcoxon = float(stats.wilcoxon(diff).pvalue)
        except ValueError:
            wilcoxon = np.nan
        rows.append(
            {
                "reference": "MOSAIC full",
                "comparator": "Early-fusion XGBoost",
                "metric": metric,
                "n_donors": n,
                "mean_difference": float(diff.mean()),
                "difference_ci95_low": float(diff.mean() - margin),
                "difference_ci95_high": float(diff.mean() + margin),
                "difference_ci95_margin": margin,
                "paired_t_pvalue": pvalue,
                "wilcoxon_pvalue": wilcoxon,
                "positive_donor_count": int((diff > 0).sum()),
                "negative_donor_count": int((diff < 0).sum()),
                "zero_donor_count": int((diff == 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def f4(value: float) -> str:
    return f"{value:.4f}"


def format_ci(row: pd.Series) -> str:
    return f"{f4(row['mean_difference'])} [{f4(row['difference_ci95_low'])}, {f4(row['difference_ci95_high'])}]"


def write_tex(stats_frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    label = {
        "accuracy": "ACC",
        "weighted_f1": "W-F1",
        "macro_f1": "M-F1",
        "balanced_accuracy": "B-ACC",
    }
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Metric & Paired difference & Paired $t$ $P$ & Wilcoxon $P$ & Positive donors \\",
        r"\midrule",
    ]
    for _, row in stats_frame.iterrows():
        lines.append(
            f"{label[row['metric']]} & {format_ci(row)} & {row['paired_t_pvalue']:.4g} & {row['wilcoxon_pvalue']:.4g} & {int(row['positive_donor_count'])}/{int(row['n_donors'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def run(args: argparse.Namespace) -> None:
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    mosaic = load_mosaic_donor_metrics(args.mosaic_dir)
    xgb = load_xgboost_donor_metrics(args.xgboost_run_level)
    stats_frame = paired_metric_rows(mosaic, xgb, METRICS[:3])
    donor_frame = mosaic.merge(xgb, on="test_donor", suffixes=("_mosaic", "_xgboost"))
    write_csv(donor_frame, out_dir / "donor_paired_metrics.csv")
    write_csv(stats_frame, out_dir / "paired_statistics.csv")
    write_csv(stats_frame, ROOT / "results/tables" / f"mosaic_n_v43_mosaic_vs_xgboost_paired_{DATE}.csv")
    write_csv(stats_frame, ROOT / "output/tables" / f"mosaic_n_v43_mosaic_vs_xgboost_paired_{DATE}.csv")
    write_tex(stats_frame, TABLE_DIR / "supplement_mosaic_vs_xgboost_paired.tex")
    print(stats_frame.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mosaic-dir", type=Path, default=DEFAULT_MOSAIC_DIR)
    parser.add_argument("--xgboost-run-level", type=Path, default=DEFAULT_XGB_RUN_LEVEL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
