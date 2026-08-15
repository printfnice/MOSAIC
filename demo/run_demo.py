#!/usr/bin/env python
"""Validate the compact ten-cell MOSAIC audit record without a checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PACKAGE = Path(__file__).resolve().parents[1]
if not (PACKAGE / "demo/inputs.npz").is_file():
    PACKAGE = Path(__file__).resolve().parents[2]
REQUIRED_COLUMNS = {
    "cell_id",
    "final_prediction",
    "final_confidence",
    "final_uncertainty",
    "rna_prediction",
    "adt_prediction",
    "fusion_prediction",
    "rna_margin",
    "adt_margin",
    "fusion_margin",
    "rna_weight",
    "adt_weight",
    "fusion_weight",
    "branch_conflict",
    "hsr_gate",
    "hsr_delta_norm",
}


def validate_record(record: pd.DataFrame, input_cell_ids: list[str]) -> None:
    """Check that the released record is a coherent audit artifact."""
    missing = sorted(REQUIRED_COLUMNS - set(record.columns))
    if missing:
        raise ValueError(f"audit record is missing columns: {missing}")
    if record["cell_id"].astype(str).tolist() != input_cell_ids:
        raise ValueError("audit record cell IDs do not match demo inputs")
    if len(record) != len(input_cell_ids):
        raise ValueError("audit record and demo input lengths differ")

    numeric = [
        "final_confidence",
        "final_uncertainty",
        "rna_margin",
        "adt_margin",
        "fusion_margin",
        "rna_weight",
        "adt_weight",
        "fusion_weight",
        "hsr_gate",
        "hsr_delta_norm",
    ]
    values = record[numeric].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("audit record contains non-finite numeric values")
    if not np.allclose(
        record["final_uncertainty"].to_numpy(float),
        1.0 - record["final_confidence"].to_numpy(float),
        atol=1e-6,
    ):
        raise ValueError("uncertainty is not the complement of confidence")
    weights = record[["rna_weight", "adt_weight", "fusion_weight"]].to_numpy(float)
    if not np.allclose(weights.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("modality weights do not sum to one")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the released ten-cell audit record. This smoke test "
            "does not perform model inference and needs no checkpoint."
        )
    )
    parser.add_argument("--package", type=Path, default=PACKAGE)
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="retained for command compatibility; validation is CPU-only",
    )
    args = parser.parse_args()
    package = args.package.resolve()
    inputs = dict(np.load(package / "demo/inputs.npz", allow_pickle=False))
    input_cell_ids = [str(value) for value in inputs["cell_id"]]
    output_path = package / "demo/audit_record_generated.csv"
    record = pd.read_csv(output_path)
    validate_record(record, input_cell_ids)
    summary = {
        "device": "cpu",
        "n_cells": int(len(record)),
        "output": str(output_path),
        "mode": "precomputed_audit_record_validation",
        "checkpoint_required": False,
    }
    print(json.dumps(summary, indent=2))
    print(record.to_string(index=False))


if __name__ == "__main__":
    main()
