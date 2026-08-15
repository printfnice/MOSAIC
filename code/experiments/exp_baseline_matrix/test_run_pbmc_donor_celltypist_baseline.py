from __future__ import annotations

import numpy as np
import pandas as pd

from run_pbmc_donor_celltypist_baseline import as_str_array, build_config, extract_celltypist_predictions, summarize


class _Result:
    def __init__(self) -> None:
        self.predicted_labels = pd.DataFrame({"predicted_labels": ["A", "B"]})
        self.probability_matrix = pd.DataFrame({"A": [0.8, 0.2], "B": [0.2, 0.8]})


def test_extract_celltypist_predictions_from_dataframe() -> None:
    y_pred, confidence = extract_celltypist_predictions(_Result())
    assert y_pred.tolist() == ["A", "B"]
    assert confidence is not None
    assert np.allclose(confidence, [0.8, 0.8])


def test_as_str_array_accepts_python_lists() -> None:
    values = as_str_array(["gene_a", "gene_b"])
    assert isinstance(values, np.ndarray)
    assert values.dtype.kind in {"U", "O"}
    assert values.tolist() == ["gene_a", "gene_b"]


def test_summarize_uses_donor_level_values() -> None:
    frame = pd.DataFrame(
        [
            {"method": "CellTypist", "test_donor": "P1", "accuracy": 0.7, "weighted_f1": 0.6, "macro_f1": 0.5, "balanced_accuracy": 0.4},
            {"method": "CellTypist", "test_donor": "P2", "accuracy": 0.9, "weighted_f1": 0.8, "macro_f1": 0.7, "balanced_accuracy": 0.6},
        ]
    )
    summary = summarize(frame)
    row = summary[summary["metric"].eq("accuracy")].iloc[0]
    assert row["n_donors"] == 2
    assert np.isclose(row["mean"], 0.8)


def test_build_config_records_published_method_and_no_cap() -> None:
    config = build_config(
        donors=["P1"],
        max_iter=200,
        n_jobs=8,
        runtime_seconds=1.0,
        celltypist_version="1.7.1",
    )
    assert config["method"] == "CellTypist"
    assert config["method_type"] == "published_single_cell_annotation"
    assert config["max_train_per_class"] == 0
    assert config["celltypist_annotation_interface"].startswith("Classifier")
    assert "standardized matrices" in config["protocol_caveat"]
