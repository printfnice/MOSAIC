from __future__ import annotations

import numpy as np
import pandas as pd

from analyze_v43_mosaic_vs_xgboost import paired_metric_rows


def test_paired_metric_rows_counts_donor_directions() -> None:
    mosaic = pd.DataFrame(
        {
            "test_donor": ["P1", "P2", "P3"],
            "accuracy": [0.8, 0.7, 0.6],
            "weighted_f1": [0.9, 0.8, 0.7],
        }
    )
    comparator = pd.DataFrame(
        {
            "test_donor": ["P1", "P2", "P3"],
            "accuracy": [0.7, 0.75, 0.55],
            "weighted_f1": [0.85, 0.85, 0.6],
        }
    )
    rows = paired_metric_rows(mosaic, comparator, ["accuracy", "weighted_f1"])
    acc = rows[rows["metric"].eq("accuracy")].iloc[0]
    assert acc["n_donors"] == 3
    assert np.isclose(acc["mean_difference"], (0.1 - 0.05 + 0.05) / 3)
    assert acc["positive_donor_count"] == 2
    assert acc["negative_donor_count"] == 1
