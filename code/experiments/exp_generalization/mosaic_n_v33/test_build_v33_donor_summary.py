import numpy as np
import pandas as pd

from build_v33_donor_summary import (
    average_per_class_seeds_by_donor,
    average_seeds_by_donor,
    build_method_summary,
    build_paired_donor_statistics,
    build_per_class_summary,
)


def _synthetic_metrics() -> pd.DataFrame:
    rows = []
    for donor, donor_shift in [("P1", 0.0), ("P2", 0.1)]:
        for seed, seed_shift in [(41, -0.01), (42, 0.0), (43, 0.01)]:
            for method, method_shift in [("reference", 0.05), ("baseline", 0.0)]:
                value = 0.7 + donor_shift + seed_shift + method_shift
                rows.append(
                    {
                        "test_donor": donor,
                        "seed": seed,
                        "method": method,
                        "accuracy": value,
                        "weighted_f1": value - 0.01,
                        "macro_f1": value - 0.10,
                        "balanced_accuracy": value - 0.08,
                    }
                )
    return pd.DataFrame(rows)


def test_average_seeds_by_donor_keeps_donor_as_statistical_unit() -> None:
    averaged = average_seeds_by_donor(_synthetic_metrics())
    assert len(averaged) == 4
    assert set(averaged["n_seeds"]) == {3}
    p1_reference = averaged[
        (averaged["test_donor"] == "P1")
        & (averaged["method"] == "reference")
    ].iloc[0]
    assert np.isclose(p1_reference["accuracy"], 0.75)


def test_paired_statistics_counts_two_donors_not_six_seeds() -> None:
    averaged = average_seeds_by_donor(_synthetic_metrics())
    paired = build_paired_donor_statistics(
        averaged,
        reference_method="reference",
    )
    row = paired[
        (paired["comparator"] == "baseline")
        & (paired["metric"] == "accuracy")
    ].iloc[0]
    assert row["n_donors"] == 2
    assert np.isclose(row["mean_difference"], 0.05)


def test_method_summary_reports_worst_donor() -> None:
    averaged = average_seeds_by_donor(_synthetic_metrics())
    summary = build_method_summary(averaged)
    row = summary[
        (summary["method"] == "reference")
        & (summary["metric"] == "accuracy")
    ].iloc[0]
    assert row["n_donors"] == 2
    assert np.isclose(row["worst_donor_value"], 0.75)


def test_per_class_summary_excludes_absent_donor_classes() -> None:
    frame = pd.DataFrame(
        {
            "test_donor": ["P1", "P1", "P2", "P2"],
            "seed": [41, 42, 41, 42],
            "method": ["reference"] * 4,
            "class_label": ["rare"] * 4,
            "f1": [0.8, 1.0, 0.0, 0.0],
            "support": [4, 4, 0, 0],
        }
    )
    donor_means = average_per_class_seeds_by_donor(frame)
    assert len(donor_means) == 1
    assert donor_means.iloc[0]["test_donor"] == "P1"
    assert np.isclose(donor_means.iloc[0]["f1"], 0.9)
    summary = build_per_class_summary(donor_means)
    assert summary.iloc[0]["n_observed_donors"] == 1
    assert summary.iloc[0]["total_test_support"] == 4
