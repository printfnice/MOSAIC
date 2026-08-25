#!/usr/bin/env python
"""Audit paired PDC101 holdout predictions without retraining either method."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as scipy_stats
import yaml
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
)


ROOT = Path(__file__).resolve().parents[3]
DATE = "2026-08-16"
BOUNDARY_CLASSES = {"cd4_cm", "cd8_cm"}


def boundary_status(label: str) -> str:
    return "central_memory" if str(label) in BOUNDARY_CLASSES else "other"


def mcnemar_exact_pvalue(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from discordant pair counts."""
    b = int(b)
    c = int(c)
    discordant = b + c
    if discordant == 0:
        return 1.0
    lower = float(scipy_stats.binom.cdf(min(b, c), discordant, 0.5))
    upper = float(scipy_stats.binom.sf(max(b, c) - 1, discordant, 0.5))
    return float(min(1.0, 2.0 * min(lower, upper)))


def validate_prediction_frames(
    mmochi: pd.DataFrame,
    mosaic: pd.DataFrame,
) -> pd.DataFrame:
    """Join exact holdout cell IDs and reject duplicate or mismatched labels."""
    required_mmochi = {
        "MMoCHi_obs_names",
        "sort_label",
        "mmochi_prediction",
        "external_holdout",
    }
    required_mosaic = {"cell_id", "sort_label", "prediction"}
    missing_mmochi = required_mmochi.difference(mmochi.columns)
    missing_mosaic = required_mosaic.difference(mosaic.columns)
    if missing_mmochi or missing_mosaic:
        raise ValueError(
            f"missing columns: MMoCHi={sorted(missing_mmochi)}, "
            f"MOSAIC={sorted(missing_mosaic)}"
        )
    mmochi = mmochi.loc[mmochi["external_holdout"].astype(bool)].copy()
    mosaic = mosaic.copy()
    if mmochi["MMoCHi_obs_names"].duplicated().any():
        raise ValueError("MMoCHi holdout IDs are not unique")
    if mosaic["cell_id"].duplicated().any():
        raise ValueError("MOSAIC prediction IDs are not unique")
    merged = mmochi.merge(
        mosaic,
        left_on="MMoCHi_obs_names",
        right_on="cell_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_mmochi", "_mosaic"),
    )
    if len(merged) != len(mmochi) or len(merged) != len(mosaic):
        raise ValueError(
            f"holdout join is incomplete: MMoCHi={len(mmochi)}, "
            f"MOSAIC={len(mosaic)}, joined={len(merged)}"
        )
    if not merged["sort_label_mmochi"].equals(merged["sort_label_mosaic"]):
        mismatches = merged.loc[
            merged["sort_label_mmochi"] != merged["sort_label_mosaic"],
            ["MMoCHi_obs_names", "sort_label_mmochi", "sort_label_mosaic"],
        ]
        raise ValueError(f"truth-label mismatch after join: {mismatches.head().to_dict('records')}")
    merged["cell_id"] = merged["MMoCHi_obs_names"].astype(str)
    merged["sort_label"] = merged["sort_label_mmochi"].astype(str)
    merged["mosaic_prediction"] = merged["prediction"].astype(str)
    merged["mmochi_prediction"] = merged["mmochi_prediction"].astype(str)
    merged["mosaic_correct"] = merged["mosaic_prediction"] == merged["sort_label"]
    merged["mmochi_correct"] = merged["mmochi_prediction"] == merged["sort_label"]
    merged["boundary_status"] = merged["sort_label"].map(boundary_status)
    merged["discordance"] = np.select(
        [
            merged["mosaic_correct"] & merged["mmochi_correct"],
            merged["mosaic_correct"] & ~merged["mmochi_correct"],
            ~merged["mosaic_correct"] & merged["mmochi_correct"],
        ],
        ["both_correct", "mosaic_only_correct", "mmochi_only_correct"],
        default="both_incorrect",
    )
    return merged


def summarize_gate_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize the saved MOSAIC ADT gate; this is not a cross-method weight test."""
    if "adt_gate" not in frame.columns:
        return pd.DataFrame(
            columns=["group", "n", "mean", "median", "std", "q25", "q75"]
        )
    rows = []
    for group, values in frame.groupby("boundary_status", sort=True)["adt_gate"]:
        values = pd.to_numeric(values, errors="coerce").dropna()
        if values.empty:
            continue
        rows.append(
            {
                "group": str(group),
                "n": int(len(values)),
                "mean": float(values.mean()),
                "median": float(values.median()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "q25": float(values.quantile(0.25)),
                "q75": float(values.quantile(0.75)),
            }
        )
    return pd.DataFrame(rows)


def _metric_row(frame: pd.DataFrame, method: str, prediction_col: str) -> dict:
    y_true = frame["sort_label"].to_numpy()
    y_pred = frame[prediction_col].to_numpy()
    labels = np.unique(y_true)
    return {
        f"{method}_accuracy": float(accuracy_score(y_true, y_pred)),
        f"{method}_balanced_accuracy": float(
            recall_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        f"{method}_weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=labels,
                average="weighted",
                zero_division=0,
            )
        ),
        f"{method}_macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_readme(out_dir: Path) -> None:
    (out_dir / "README.md").write_text(
        """# V8.2 PDC101 paired prediction audit

This artifact compares the valid no-HTO MOSAIC-HPM-lite prediction file with
the official MMoCHi external-holdout prediction file on exactly the same
PDC101 cells. It performs a one-to-one cell-ID join, checks truth-label
agreement, and reports paired correctness counts and an exact two-sided
McNemar test.

The saved `adt_gate` is summarized within MOSAIC by boundary class and is not
treated as directly comparable to MMoCHi certainty. A cross-method modality
weight test is therefore marked `not_estimable`; no weight is reconstructed
from final predictions. Both source runs exclude HTO features as model input.
This is an audit of existing held-out predictions and does not retrain either
method or select a threshold using the holdout labels.
""",
        encoding="utf-8",
    )


def _write_yaml(path: Path, config: dict) -> None:
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")


def run_audit(
    mmochi_path: Path,
    mosaic_path: Path,
    out_dir: Path,
    validate_only: bool = False,
) -> dict:
    mmochi = pd.read_csv(mmochi_path)
    mosaic = pd.read_csv(mosaic_path)
    merged = validate_prediction_frames(mmochi, mosaic)
    metadata_path = mmochi_path.parent / "metadata.json"
    mosaic_metadata_path = mosaic_path.parent / "metadata.json"
    mmochi_metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    mosaic_metadata = (
        json.loads(mosaic_metadata_path.read_text())
        if mosaic_metadata_path.exists()
        else {}
    )
    preflight = {
        "status": "ready",
        "n_mmochi_external_holdout": int(mmochi["external_holdout"].astype(bool).sum()),
        "n_mosaic_rows": int(len(mosaic)),
        "n_joined_pairs": int(len(merged)),
        "truth_labels_match": True,
        "mmochi_source": str(mmochi_path),
        "mosaic_source": str(mosaic_path),
        "mmochi_metadata": mmochi_metadata,
        "mosaic_metadata": mosaic_metadata,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "preflight.json").write_text(json.dumps(preflight, indent=2), encoding="utf-8")
    if validate_only:
        return preflight

    b = int((merged["mosaic_correct"] & ~merged["mmochi_correct"]).sum())
    c = int((~merged["mosaic_correct"] & merged["mmochi_correct"]).sum())
    discordant = b + c
    mcnemar_p = mcnemar_exact_pvalue(b, c)
    mcnemar_chi2 = float(((abs(b - c) - 1) ** 2) / discordant) if discordant else 0.0
    summary = {
        "status": "ready",
        "n_joined_pairs": int(len(merged)),
        "dataset": "GSE229791 PDC101 sorted external holdout",
        "n_pairs": int(len(merged)),
        "mosaic_method": str(mosaic_metadata.get("method", "MOSAIC-HPM-lite")),
        "mmochi_method": "MMoCHi official prediction",
        "mosaic_correct_mmochi_wrong": b,
        "mmochi_correct_mosaic_wrong": c,
        "discordant_pairs": discordant,
        "mcnemar_exact_p_two_sided": mcnemar_p,
        "mcnemar_chi2_continuity": mcnemar_chi2,
        "paired_accuracy_delta_mosaic_minus_mmochi": float(
            (b - c) / len(merged)
        ),
        "boundary_classes": sorted(BOUNDARY_CLASSES),
        "boundary_n": int((merged["boundary_status"] == "central_memory").sum()),
        "weight_comparison_status": "not_estimable",
        "weight_comparison_reason": (
            "MOSAIC adt_gate and MMoCHi certainty are different estimands; "
            "MMoCHi modality weights were not saved in the evaluated artifact"
        ),
        **_metric_row(merged, "mosaic", "mosaic_prediction"),
        **_metric_row(merged, "mmochi", "mmochi_prediction"),
    }
    pd.DataFrame([summary]).to_csv(out_dir / "results_summary.csv", index=False)
    output_columns = [
        "cell_id",
        "sort_label",
        "mosaic_prediction",
        "mmochi_prediction",
        "mosaic_correct",
        "mmochi_correct",
        "discordance",
        "boundary_status",
    ]
    for optional in ("adt_gate", "mmochi_certainty"):
        if optional in merged:
            output_columns.append(optional)
    merged[output_columns].sort_values("cell_id").to_csv(
        out_dir / "per_cell_paired_outcomes.csv",
        index=False,
    )
    summarize_gate_distribution(merged).to_csv(
        out_dir / "gate_distribution.csv",
        index=False,
    )
    confidence_rows = []
    for method, column in (("MOSAIC_adt_gate", "adt_gate"), ("MMoCHi_certainty", "mmochi_certainty")):
        if column not in merged:
            continue
        values = pd.to_numeric(merged[column], errors="coerce").dropna()
        confidence_rows.append(
            {
                "method": method,
                "n": int(len(values)),
                "mean": float(values.mean()),
                "median": float(values.median()),
                "q25": float(values.quantile(0.25)),
                "q75": float(values.quantile(0.75)),
                "estimand": "method-specific confidence/gate; not a paired weight comparison",
            }
        )
    pd.DataFrame(confidence_rows).to_csv(out_dir / "confidence_distribution.csv", index=False)
    tests = {
        "paired_test": {
            "test": "exact two-sided McNemar",
            "b_mosaic_only_correct": b,
            "c_mmochi_only_correct": c,
            "discordant_pairs": discordant,
            "p_value": mcnemar_p,
            "continuity_corrected_chi2": mcnemar_chi2,
        },
        "boundary_definition": {
            "status": "predeclared",
            "classes": sorted(BOUNDARY_CLASSES),
            "rule": "sort_label in {cd4_cm, cd8_cm}",
        },
        "gate_distribution": {
            "status": "estimable_within_MOSAIC_only" if "adt_gate" in merged else "not_estimable",
            "field": "adt_gate" if "adt_gate" in merged else None,
            "interpretation": "saved MOSAIC ADT gate summarized by boundary status; not a cross-method paired test",
        },
        "cross_method_modality_weight": {
            "status": "not_estimable",
            "reason": summary["weight_comparison_reason"],
        },
    }
    (out_dir / "statistical_tests.json").write_text(
        json.dumps(tests, indent=2),
        encoding="utf-8",
    )
    (out_dir / "split_seed_metadata.json").write_text(
        json.dumps(
            {
                "dataset": "GSE229791 PDC101 sorted external holdout",
                "split_unit": "precomputed external_holdout=True rows",
                "n_pairs": int(len(merged)),
                "mmochi_source_metadata": str(metadata_path),
                "mosaic_source_metadata": str(mosaic_metadata_path),
                "mmochi_metadata": mmochi_metadata,
                "mosaic_metadata": mosaic_metadata,
                "hto_policy": "excluded from both evaluated model inputs; cell IDs may retain HTO-derived source suffixes",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "source_checksums.json").write_text(
        json.dumps(
            {
                "mmochi_predictions_sha256": _sha256(mmochi_path),
                "mosaic_predictions_sha256": _sha256(mosaic_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mmochi-predictions",
        type=Path,
        default=Path(
            "results/exp_generalization/mmochi_pdc101_sorted_ext_holdout_thresholds/holdout_predictions.csv"
        ),
    )
    parser.add_argument(
        "--mosaic-predictions",
        type=Path,
        default=Path(
            "results/exp_generalization/mosaic2_pdc101_hpm_lite_best_no_hto/predictions.csv"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "results/experiments/v8.2_missing_modality_pdc_audit/pdc101_paired_audit"
        ),
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    paths = {
        key: value if value.is_absolute() else ROOT / value
        for key, value in {
            "mmochi": args.mmochi_predictions,
            "mosaic": args.mosaic_predictions,
            "out": args.out_dir,
        }.items()
    }
    config = {
        "date": DATE,
        "command": " ".join(sys.argv),
        "mmochi_predictions": str(paths["mmochi"]),
        "mosaic_predictions": str(paths["mosaic"]),
        "join_key": "MMoCHi_obs_names == cell_id",
        "holdout_policy": "MMoCHi external_holdout=True; exact one-to-one join",
        "paired_test": "exact two-sided McNemar",
        "boundary_classes": sorted(BOUNDARY_CLASSES),
        "cross_method_modality_weight_status": "not_estimable",
    }
    paths["out"].mkdir(parents=True, exist_ok=True)
    _write_yaml(paths["out"] / "config.yaml", config)
    _write_readme(paths["out"])
    summary = run_audit(
        paths["mmochi"],
        paths["mosaic"],
        paths["out"],
        validate_only=args.validate_only,
    )
    log_lines = [
        f"status={summary['status']}",
        f"n_joined_pairs={summary['n_joined_pairs']}",
        f"validate_only={args.validate_only}",
    ]
    if not args.validate_only:
        log_lines.extend(
            [
                f"mcnemar_exact_p_two_sided={summary['mcnemar_exact_p_two_sided']}",
                f"paired_accuracy_delta={summary['paired_accuracy_delta_mosaic_minus_mmochi']}",
                "weight_comparison_status=not_estimable",
            ]
        )
    (paths["out"] / "run.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    if not args.validate_only:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
