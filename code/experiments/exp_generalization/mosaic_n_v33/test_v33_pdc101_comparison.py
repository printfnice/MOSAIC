import numpy as np
import pandas as pd
import pytest

from build_v33_pdc101_comparison import (
    _bootstrap_mmochi_per_class,
    summarize_mosaic_per_class,
    summarize_mosaic_seeds,
)
from run_v33_pdc101_mosaic_n import (
    PDC_HSR_GROUPS,
    _dense_float32,
    allowed_protein_mask,
    split_train_validation,
    validate_locked_holdout,
)


def test_allowed_protein_mask_excludes_hto_and_controls() -> None:
    names = [
        "HTO1",
        "Mouse_IgG1_isotype_Ctrl",
        "Rat-IgG2b",
        "CD3",
        "CD45RA",
    ]
    assert allowed_protein_mask(names).tolist() == [False, False, False, True, True]


def test_processed_pdc_values_are_not_log_transformed_again() -> None:
    values = np.array([[0.0, 1.5, 6.9]], dtype=np.float32)
    transformed = _dense_float32(values)
    assert np.array_equal(transformed, values)


def test_split_train_validation_never_uses_locked_holdout() -> None:
    labels = np.array(["a", "a", "a", "b", "b", "b", "a", "b"])
    holdout = np.array([False, False, False, False, False, False, True, True])
    train_idx, val_idx, test_idx = split_train_validation(
        labels,
        holdout,
        seed=42,
        validation_fraction=0.2,
    )
    assert set(test_idx) == {6, 7}
    assert not set(train_idx) & set(test_idx)
    assert not set(val_idx) & set(test_idx)
    assert set(train_idx) | set(val_idx) | set(test_idx) == set(range(len(labels)))


def test_locked_holdout_requires_external_flag_and_matching_truth() -> None:
    cell_ids = np.array(["a", "b", "c"])
    labels = np.array(["x", "y", "z"])
    holdout = pd.DataFrame(
        {
            "MMoCHi_obs_names": ["b", "c"],
            "sort_label": ["y", "z"],
            "external_holdout": [True, True],
        }
    )
    assert validate_locked_holdout(cell_ids, labels, holdout).tolist() == [
        False,
        True,
        True,
    ]
    invalid = holdout.copy()
    invalid.loc[1, "sort_label"] = "wrong"
    with pytest.raises(ValueError, match="truth labels"):
        validate_locked_holdout(cell_ids, labels, invalid)


def test_pdc_hsr_groups_cover_cd4_and_cd8_boundaries() -> None:
    assert PDC_HSR_GROUPS == [
        ["cd4_n", "cd4_cm", "cd4_em"],
        ["cd8_n", "cd8_cm", "cd8_em", "cd8_emra"],
    ]


def test_summarize_mosaic_seeds_reports_seed_interval() -> None:
    frame = pd.DataFrame(
        {
            "seed": [41, 42, 43],
            "accuracy": [0.90, 0.91, 0.92],
            "weighted_f1": [0.89, 0.90, 0.91],
            "macro_f1": [0.88, 0.89, 0.90],
        }
    )
    summary = summarize_mosaic_seeds(frame)
    accuracy = summary[summary["metric"] == "accuracy"].iloc[0]
    assert accuracy["n_seeds"] == 3
    assert np.isclose(accuracy["mean"], 0.91)
    assert accuracy["ci95_margin"] > 0


def test_pdc_per_class_summary_preserves_seed_and_support_units() -> None:
    frame = pd.DataFrame(
        {
            "seed": [41, 42, 43, 41, 42, 43],
            "class_label": ["a", "a", "a", "b", "b", "b"],
            "f1": [0.8, 0.9, 1.0, 0.6, 0.7, 0.8],
            "support": [10, 10, 10, 20, 20, 20],
        }
    )
    summary = summarize_mosaic_per_class(frame)
    row = summary[summary["class_label"].eq("a")].iloc[0]
    assert row["n_seeds"] == 3
    assert row["support"] == 10
    assert np.isclose(row["mean_f1"], 0.9)


def test_mmochi_per_class_bootstrap_reports_all_truth_labels() -> None:
    predictions = pd.DataFrame(
        {
            "sort_label": ["a", "a", "b", "b"],
            "mmochi_prediction": ["a", "b", "b", "b"],
        }
    )
    summary = _bootstrap_mmochi_per_class(
        predictions,
        n_bootstrap=20,
        seed=42,
    )
    assert set(summary["class_label"]) == {"a", "b"}
    assert set(summary["support"]) == {2}
    assert summary["ci95_margin"].ge(0).all()
