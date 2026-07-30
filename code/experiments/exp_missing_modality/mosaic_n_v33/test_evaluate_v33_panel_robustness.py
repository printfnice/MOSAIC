import numpy as np

from evaluate_v33_panel_robustness import (
    build_random_feature_mask,
    expected_calibration_error,
    marker_feature_mask,
    multiclass_brier_score,
)


def test_random_feature_mask_is_deterministic_and_feature_level() -> None:
    first = build_random_feature_mask(20, fraction=0.5, seed=42)
    second = build_random_feature_mask(20, fraction=0.5, seed=42)
    different = build_random_feature_mask(20, fraction=0.5, seed=43)

    assert first.dtype == bool
    assert first.sum() == 10
    assert np.array_equal(first, second)
    assert not np.array_equal(first, different)


def test_marker_feature_mask_matches_declared_alias_prefixes() -> None:
    names = ["CD3-1", "CD3-2", "CD4-1", "CD45RA", "CD19"]
    mask = marker_feature_mask(names, ["CD3", "CD4"])
    assert mask.tolist() == [True, True, True, False, False]


def test_calibration_metrics_are_zero_for_perfect_predictions() -> None:
    y_true = np.array([0, 1, 0])
    probabilities = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ]
    )
    assert np.isclose(expected_calibration_error(y_true, probabilities), 0.0)
    assert np.isclose(multiclass_brier_score(y_true, probabilities), 0.0)

