#!/usr/bin/env python
"""Run the checkpoint-free ten-cell MOSAIC audit-validation demonstration.

This release demo validates the packaged precomputed audit record against the
packaged test inputs and writes a deterministic output copy plus a JSON
summary. It is intentionally not a trained-model inference or performance
reproduction command because checkpoints are excluded from the public package.
"""

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


def validate_inputs(inputs: dict[str, np.ndarray]) -> list[str]:
    """Validate the packaged test-data schema and return its cell IDs."""
    required = {"gene", "protein", "cell_id"}
    missing = sorted(required - set(inputs))
    if missing:
        raise ValueError(f"demo inputs are missing arrays: {missing}")
    gene = np.asarray(inputs["gene"])
    protein = np.asarray(inputs["protein"])
    if gene.ndim != 2 or gene.shape[1] != 3000:
        raise ValueError(f"gene must have shape (n_cells, 3000), got {gene.shape}")
    if protein.ndim != 2 or protein.shape[1] != 224:
        raise ValueError(
            f"protein must have shape (n_cells, 224), got {protein.shape}"
        )
    if gene.shape[0] != protein.shape[0]:
        raise ValueError("gene and protein cell counts differ")
    if not np.isfinite(gene).all() or not np.isfinite(protein).all():
        raise ValueError("demo inputs contain non-finite values")
    cell_ids = [str(value) for value in np.asarray(inputs["cell_id"])]
    if len(cell_ids) != gene.shape[0] or len(set(cell_ids)) != len(cell_ids):
        raise ValueError("demo cell IDs do not match the input matrices")
    return cell_ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the released ten-cell test data and audit record. This "
            "checkpoint-free demo writes a validated CSV and JSON summary; it "
            "does not perform trained-model inference."
        )
    )
    parser.add_argument("--package", type=Path, default=PACKAGE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="directory for generated demo outputs (default: <package>/demo/output)",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="retained for command compatibility; validation is CPU-only",
    )
    args = parser.parse_args()
    package = args.package.resolve()
    input_path = package / "demo/inputs.npz"
    inputs = dict(np.load(input_path, allow_pickle=False))
    input_cell_ids = validate_inputs(inputs)
    source_record_path = package / "demo/audit_record_generated.csv"
    record = pd.read_csv(source_record_path)
    validate_record(record, input_cell_ids)
    output_dir = (args.output_dir or package / "demo/output").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_record_path = output_dir / "audit_record_validated.csv"
    record.to_csv(output_record_path, index=False)
    summary = {
        "device": "cpu",
        "n_cells": int(len(record)),
        "input": str(input_path),
        "input_shapes": {
            "gene": list(np.asarray(inputs["gene"]).shape),
            "protein": list(np.asarray(inputs["protein"]).shape),
        },
        "output": str(output_record_path),
        "mode": "precomputed_audit_record_validation",
        "checkpoint_required": False,
        "scientific_inference": False,
    }
    summary_path = output_dir / "demo_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    summary["summary"] = str(summary_path)
    print(json.dumps(summary, indent=2))
    print(record.to_string(index=False))


if __name__ == "__main__":
    main()
