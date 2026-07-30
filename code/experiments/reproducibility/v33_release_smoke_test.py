#!/usr/bin/env python
"""Validate the synthetic V33 release fixture with train-only preprocessing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Sequence

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


REQUIRED_METADATA = {"cell_id", "donor", "split", "label"}
EXPECTED_SPLITS = {"train", "val", "test"}
PACKAGED_FIXTURE = Path(
    "testdata/experiments/reproducibility/testdata/v33_synthetic_smoke.csv"
)


def run_smoke(fixture_path: Path) -> Dict[str, object]:
    """Fit a tiny deterministic classifier and audit split isolation."""

    frame = pd.read_csv(fixture_path)
    missing = sorted(REQUIRED_METADATA - set(frame.columns))
    if missing:
        raise ValueError("missing fixture columns: " + ", ".join(missing))
    feature_columns = [
        column
        for column in frame.columns
        if column.startswith("rna_") or column.startswith("adt_")
    ]
    if not feature_columns:
        raise ValueError("fixture contains no RNA/ADT feature columns")
    if set(frame["split"].astype(str)) != EXPECTED_SPLITS:
        raise ValueError("fixture must contain train, val and test splits")
    if frame[feature_columns].isna().any().any():
        raise ValueError("fixture feature matrix contains missing values")

    donors = {
        split: set(
            frame.loc[frame["split"] == split, "donor"].astype(str)
        )
        for split in sorted(EXPECTED_SPLITS)
    }
    if (
        donors["train"] & donors["val"]
        or donors["train"] & donors["test"]
        or donors["val"] & donors["test"]
    ):
        raise ValueError("fixture donors overlap across train/val/test")

    train = frame["split"].eq("train")
    validation = frame["split"].eq("val")
    test = frame["split"].eq("test")
    if frame.loc[train, "label"].nunique() < 2:
        raise ValueError("training fixture must contain at least two labels")

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(random_state=0, max_iter=500),
    )
    model.fit(frame.loc[train, feature_columns], frame.loc[train, "label"])
    validation_prediction = model.predict(
        frame.loc[validation, feature_columns]
    )
    test_prediction = model.predict(frame.loc[test, feature_columns])

    return {
        "fixture": str(fixture_path),
        "n_cells": int(len(frame)),
        "n_features": int(len(feature_columns)),
        "train_donors": sorted(donors["train"]),
        "validation_donors": sorted(donors["val"]),
        "test_donors": sorted(donors["test"]),
        "validation_accuracy": float(
            accuracy_score(frame.loc[validation, "label"], validation_prediction)
        ),
        "test_accuracy": float(
            accuracy_score(frame.loc[test, "label"], test_prediction)
        ),
        "test_macro_f1": float(
            f1_score(
                frame.loc[test, "label"],
                test_prediction,
                average="macro",
                zero_division=0,
            )
        ),
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-dir", type=Path)
    parser.add_argument("--fixture", type=Path)
    args = parser.parse_args(argv)
    if bool(args.release_dir) == bool(args.fixture):
        parser.error("provide exactly one of --release-dir or --fixture")
    fixture = (
        args.fixture
        if args.fixture is not None
        else args.release_dir / PACKAGED_FIXTURE
    )
    result = run_smoke(fixture.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
