#!/usr/bin/env python
"""Evaluate transfer of the locked audit policy to the GSE164378 5P assay cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score


ROOT = Path(__file__).resolve().parents[3]
DATE = "2026-08-16"
MAIN_DONORS = {f"P{i}" for i in range(1, 9)}


def validate_external_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "cell_id",
        "mask_condition",
        "label",
        "true_l1",
        "prediction",
        "pred_l1",
        "decision",
        "confidence",
        "parent_safe",
        "external_label_used_for_policy_selection",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing external prediction columns: {sorted(missing)}")
    full = frame.loc[frame["mask_condition"].eq("full")].copy()
    if full.empty:
        raise ValueError("no full mask_condition rows found")
    if full["cell_id"].duplicated().any():
        raise ValueError("full external prediction cell IDs are not unique")
    policy_flags = full["external_label_used_for_policy_selection"].astype(str).str.lower()
    if not policy_flags.eq("no").all():
        raise ValueError("external labels were used for policy selection")
    full["parent_safe"] = full["parent_safe"].astype(bool)
    return full


def classify_cohort_identity(
    dataset_name: str,
    observed_donors: set[str],
    main_donors: set[str],
) -> dict[str, object]:
    same_donors = observed_donors == main_donors
    same_study = "GSE164378" in dataset_name
    independent = not (same_donors and same_study)
    return {
        "dataset_name": dataset_name,
        "observed_donors": sorted(observed_donors),
        "main_protocol_donors": sorted(main_donors),
        "same_donor_set": same_donors,
        "same_study_indicator": same_study,
        "independent_external_cohort": independent,
        "transfer_type": (
            "independent_external_cohort"
            if independent
            else "same-study assay-cohort replication"
        ),
        "scope_caveat": (
            "not independent donor validation; same GSE164378 study and donor set"
            if not independent
            else "independence requires separate-study/protocol verification"
        ),
    }


def _safe_rate(values: pd.Series) -> float:
    return float(values.mean()) if len(values) else float("nan")


def summarize_transfer(frame: pd.DataFrame) -> dict[str, object]:
    accepted = frame["decision"].astype(str).str.lower().eq("l3_accept")
    fallback = ~accepted
    accepted_frame = frame.loc[accepted]
    fallback_frame = frame.loc[fallback]
    mean_confidence = (
        float(pd.to_numeric(frame["confidence"], errors="coerce").mean())
        if "confidence" in frame
        else float("nan")
    )
    return {
        "n_eval": int(len(frame)),
        "n_accepted": int(accepted.sum()),
        "n_fallback": int(fallback.sum()),
        "accept_rate": float(accepted.mean()),
        "l3_accuracy_all": float(
            accuracy_score(frame["label"], frame["prediction"])
        ),
        "accepted_l3_accuracy": (
            float(accuracy_score(accepted_frame["label"], accepted_frame["prediction"]))
            if len(accepted_frame)
            else float("nan")
        ),
        "parent_safe_rate": _safe_rate(frame["parent_safe"]),
        "unsafe_wrong_parent_rate": 1.0 - _safe_rate(frame["parent_safe"]),
        "fallback_l1_error_rate_raw": (
            float((fallback_frame["true_l1"] != fallback_frame["pred_l1"]).mean())
            if len(fallback_frame)
            else float("nan")
        ),
        "mean_confidence": mean_confidence,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_yaml(path: Path, config: dict) -> None:
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _write_readme(out_dir: Path) -> None:
    (out_dir / "README.md").write_text(
        """# V8.2 GSE164378 5P audit transfer

This artifact transfers the already locked, validation-only MOSAIC audit
decision to the GSE164378 5P assay cohort using the existing full-panel
external prediction records. It does not tune a threshold or policy on the
external labels. The labels are used only for retrospective evaluation.

The cohort contains the same P1--P8 donors and belongs to the same GSE164378
study as the PBMC protocol. It is therefore reported as same-study
assay-cohort replication, not independent donor validation. The artifact is
useful for checking protocol transfer, but it does not expand the independent
generalization claim.
""",
        encoding="utf-8",
    )


def _load_donor_map(h5ad_path: Path) -> tuple[pd.Series, dict[str, object]]:
    dataset = ad.read_h5ad(h5ad_path, backed="r")
    if "donor" not in dataset.obs.columns:
        raise ValueError("GSE164378 h5ad lacks the donor column")
    donor = dataset.obs["donor"].astype(str)
    metadata = {
        "n_cells": int(dataset.n_obs),
        "n_features": int(dataset.n_vars),
        "obs_columns": list(dataset.obs.columns),
        "donor_values": sorted(donor.unique().tolist()),
        "label_columns_present": [
            column
            for column in ("celltype.l1", "celltype.l2", "celltype.l3")
            if column in dataset.obs.columns
        ],
    }
    return donor, metadata


def run_transfer(
    prediction_path: Path,
    h5ad_path: Path,
    calibration_path: Path,
    out_dir: Path,
    validate_only: bool = False,
) -> dict[str, object]:
    predictions = validate_external_predictions(pd.read_csv(prediction_path))
    donor_map, dataset_metadata = _load_donor_map(h5ad_path)
    missing_ids = sorted(set(predictions["cell_id"]) - set(donor_map.index))
    if missing_ids:
        raise ValueError(f"prediction IDs absent from h5ad: {missing_ids[:3]}")
    predictions["donor"] = predictions["cell_id"].map(donor_map)
    if predictions["donor"].isna().any():
        raise ValueError("donor mapping produced missing values")
    cohort_identity = classify_cohort_identity(
        "GSE164378 5P",
        set(predictions["donor"].astype(str)),
        MAIN_DONORS,
    )
    calibration = pd.read_csv(calibration_path)
    calibration_policy_sources = (
        calibration.get("policy_source", pd.Series(dtype=str)).astype(str).unique().tolist()
    )
    if calibration_policy_sources and set(calibration_policy_sources) != {"validation_only"}:
        raise ValueError(f"calibration policy is not validation-only: {calibration_policy_sources}")
    preflight = {
        "status": "ready_with_scope_caveat",
        "prediction_rows_full": int(len(predictions)),
        "prediction_ids_match_h5ad": True,
        "dataset_metadata": dataset_metadata,
        "cohort_identity": cohort_identity,
        "calibration_policy_path": str(calibration_path),
        "calibration_policy_sources": calibration_policy_sources,
        "external_label_used_for_policy_selection": "no",
        "transfer_estimand": "locked full-panel audit decision and parent-safe outcome",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "preflight.json").write_text(json.dumps(preflight, indent=2), encoding="utf-8")
    pd.DataFrame([cohort_identity]).to_csv(out_dir / "cohort_identity_audit.csv", index=False)
    if validate_only:
        return preflight

    summary = {
        "status": preflight["status"],
        "transfer_type": cohort_identity["transfer_type"],
        "independent_external_cohort": cohort_identity["independent_external_cohort"],
        **summarize_transfer(predictions),
    }
    pd.DataFrame([summary]).to_csv(out_dir / "results_summary.csv", index=False)
    records = predictions[
        [
            "cell_id",
            "donor",
            "label",
            "true_l1",
            "prediction",
            "pred_l1",
            "decision",
            "confidence",
            "parent_safe",
            "external_label_used_for_policy_selection",
        ]
    ].copy()
    records["l3_correct"] = records["label"] == records["prediction"]
    records["fallback_l1_error"] = (
        records["decision"].ne("l3_accept")
        & records["true_l1"].ne(records["pred_l1"])
    )
    records.to_csv(out_dir / "audit_records.csv", index=False)

    donor_rows = []
    for donor, group in predictions.groupby("donor", sort=True):
        donor_rows.append({"donor": str(donor), **summarize_transfer(group)})
    pd.DataFrame(donor_rows).to_csv(out_dir / "per_donor_summary.csv", index=False)

    class_rows = []
    for label, group in predictions.groupby("label", sort=True):
        class_rows.append(
            {
                "label": str(label),
                "n": int(len(group)),
                "l3_accuracy": float((group["label"] == group["prediction"]).mean()),
                "parent_safe_rate": _safe_rate(group["parent_safe"]),
                "accept_rate": float(group["decision"].eq("l3_accept").mean()),
            }
        )
    pd.DataFrame(class_rows).to_csv(out_dir / "per_class_summary.csv", index=False)
    (out_dir / "split_seed_metadata.json").write_text(
        json.dumps(
            {
                "dataset": "GSE164378 5P",
                "split_unit": "existing full-panel external prediction artifact",
                "prediction_source": str(prediction_path),
                "h5ad_source": str(h5ad_path),
                "calibration_policy_source": str(calibration_path),
                "policy_selection": "validation_only; external labels not used",
                "cohort_identity": cohort_identity,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "source_checksums.json").write_text(
        json.dumps(
            {
                "prediction_sha256": _sha256(prediction_path),
                "h5ad_sha256": _sha256(h5ad_path),
                "calibration_policy_sha256": _sha256(calibration_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path(
            "results/exp_generalization/original_mosaic_strict/v10_r61_mosaic_n_gse164378_adt_guard_full_seed42/external_predictions.csv"
        ),
    )
    parser.add_argument(
        "--h5ad",
        type=Path,
        default=Path("data/GSE164378/gse164378_5p_protein.h5ad"),
    )
    parser.add_argument(
        "--calibration-policy",
        type=Path,
        default=Path(
            "results/exp_generalization/original_mosaic_strict/v10_mosaic_n_gse164378_adt_guard_seed42/calibration_policy.csv"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "results/experiments/v8.2_missing_modality_pdc_audit/secondary_cohort"
        ),
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    paths = {
        key: value if value.is_absolute() else ROOT / value
        for key, value in {
            "predictions": args.predictions,
            "h5ad": args.h5ad,
            "calibration": args.calibration_policy,
            "out": args.out_dir,
        }.items()
    }
    paths["out"].mkdir(parents=True, exist_ok=True)
    _write_yaml(
        paths["out"] / "config.yaml",
        {
            "date": DATE,
            "command": " ".join(sys.argv),
            "predictions": str(paths["predictions"]),
            "h5ad": str(paths["h5ad"]),
            "calibration_policy": str(paths["calibration"]),
            "mask_condition": "full",
            "policy_selection": "validation_only; external labels not used",
            "main_protocol_donors": sorted(MAIN_DONORS),
            "claim_boundary": "same-study assay-cohort replication, not independent external validation",
        },
    )
    _write_readme(paths["out"])
    summary = run_transfer(
        paths["predictions"],
        paths["h5ad"],
        paths["calibration"],
        paths["out"],
        validate_only=args.validate_only,
    )
    lines = [
        f"status={summary['status']}",
        f"transfer_type={summary['cohort_identity']['transfer_type'] if args.validate_only else summary['transfer_type']}",
        f"validate_only={args.validate_only}",
    ]
    if not args.validate_only:
        lines.extend(
            [
                f"n_eval={summary['n_eval']}",
                f"accept_rate={summary['accept_rate']}",
                f"parent_safe_rate={summary['parent_safe_rate']}",
            ]
        )
    (paths["out"] / "run.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not args.validate_only:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
