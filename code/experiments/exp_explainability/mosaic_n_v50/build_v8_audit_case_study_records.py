#!/usr/bin/env python
"""Materialize the three current audit case records with provenance hashes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
DATE = "2026-08-14"
SOURCE = ROOT / "results/exp_explainability/mosaic_n_v34/attribution_faithfulness/audit_case_studies.csv"

ROLES = {
    "agreement_correct": "positive agreement record",
    "modality_conflict": "modality-conflict review record",
    "hsr_high_delta_or_changed": "bounded HSR action record",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build() -> pd.DataFrame:
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    frame = pd.read_csv(SOURCE)
    required = {
        "case_type",
        "test_donor",
        "seed",
        "cell_id",
        "label",
        "prediction",
        "branch_disagreement",
        "prediction_changed_by_hsr",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"case artifact missing columns: {sorted(missing)}")
    if set(frame["case_type"]) != set(ROLES):
        raise ValueError("case artifact does not contain exactly the three declared records")
    if frame["cell_id"].duplicated().any():
        raise ValueError("case artifact contains duplicate cell identifiers")
    frame["case_role"] = frame["case_type"].map(ROLES)
    frame["source_artifact"] = str(SOURCE.relative_to(ROOT))
    frame["source_sha256"] = sha256(SOURCE)
    frame["public_artifact"] = "N (local release unless uploaded)"
    frame["interpretation_boundary"] = (
        "Structured review evidence; not an automatic correctness guarantee"
    )
    return frame.sort_values("case_type").reset_index(drop=True)


def main() -> None:
    frame = build()
    output_dir = ROOT / "results/tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"mosaic_n_v8_audit_case_study_records_{DATE}.csv"
    frame.to_csv(output, index=False)
    (ROOT / "output/tables").mkdir(parents=True, exist_ok=True)
    frame.to_csv(ROOT / "output/tables" / output.name, index=False)
    metadata = {
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": frame["source_sha256"].iloc[0],
        "n_records": int(len(frame)),
        "roles": ROLES,
        "public_artifact": False,
    }
    (output_dir / f"mosaic_n_v8_audit_case_study_records_{DATE}.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(frame[["case_type", "test_donor", "seed", "cell_id", "case_role"]].to_string(index=False))


if __name__ == "__main__":
    main()
