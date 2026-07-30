from pathlib import Path

import numpy as np
import pandas as pd

from build_pbmc_nested_donor_cache import (
    DONOR_ORDER,
    assign_nested_splits,
    build_label_support,
    fixed_validation_donor,
    validate_nested_arrays,
)


def test_fixed_validation_donor_uses_predeclared_cycle() -> None:
    observed = [fixed_validation_donor(donor) for donor in DONOR_ORDER]
    assert observed == ["P2", "P3", "P4", "P5", "P6", "P7", "P8", "P1"]


def test_assign_nested_splits_holds_out_exact_donors() -> None:
    donors = np.array(["P1", "P1", "P2", "P3", "P8"])
    splits = assign_nested_splits(donors, test_donor="P8", validation_donor="P1")

    assert splits.tolist() == ["val", "val", "train", "train", "test"]
    assert set(donors[splits == "test"]) == {"P8"}
    assert set(donors[splits == "val"]) == {"P1"}
    assert set(donors[splits == "train"]) == {"P2", "P3"}


def test_build_label_support_marks_natural_unknowns() -> None:
    labels = np.array(["A", "A", "B", "C", "D", "D"])
    splits = np.array(["train", "val", "test", "test", "test", "train"])
    support = build_label_support(labels, splits).set_index("class_label")

    assert bool(support.loc["A", "known_to_train"])
    assert not bool(support.loc["B", "known_to_train"])
    assert bool(support.loc["B", "natural_unknown_in_test"])
    assert bool(support.loc["C", "natural_unknown_in_test"])
    assert not bool(support.loc["D", "natural_unknown_in_test"])


def test_validate_nested_arrays_rejects_donor_leakage() -> None:
    arrays = {
        "gene": np.zeros((4, 2), dtype=np.float32),
        "protein": np.zeros((4, 1), dtype=np.float32),
        "labels": np.array([0, 1, 0, 1]),
        "donors": np.array(["P1", "P2", "P3", "P4"]),
        "train_idx": np.array([0, 1]),
        "val_idx": np.array([2]),
        "test_idx": np.array([3]),
        "cell_ids": np.array(["a", "b", "c", "d"]),
    }
    validate_nested_arrays(arrays, test_donor="P4", validation_donor="P3")

    arrays["train_idx"] = np.array([0, 3])
    arrays["test_idx"] = np.array([1])
    try:
        validate_nested_arrays(arrays, test_donor="P4", validation_donor="P3")
    except ValueError as exc:
        assert "test donor leaked" in str(exc)
    else:
        raise AssertionError("expected donor leakage to be rejected")


def test_manifest_has_one_row_per_cell_and_locked_split(tmp_path: Path) -> None:
    donors = np.array(["P1", "P2", "P3"])
    splits = assign_nested_splits(donors, test_donor="P3", validation_donor="P2")
    frame = pd.DataFrame({"cell_id": ["a", "b", "c"], "donor": donors, "split": splits})
    path = tmp_path / "manifest.csv"
    frame.to_csv(path, index=False)

    loaded = pd.read_csv(path)
    assert loaded["cell_id"].is_unique
    assert loaded.set_index("donor")["split"].to_dict() == {
        "P1": "train",
        "P2": "val",
        "P3": "test",
    }
