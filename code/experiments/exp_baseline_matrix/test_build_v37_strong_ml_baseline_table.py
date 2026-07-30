from __future__ import annotations

from build_v37_strong_ml_baseline_table import METHOD_ORDER, build_rows


def test_build_rows_formats_metric_cells() -> None:
    summary = {}
    for metric, mean in [
        ("accuracy", "0.8"),
        ("weighted_f1", "0.7"),
        ("macro_f1", "0.6"),
        ("balanced_accuracy", "0.5"),
    ]:
        summary[("RNA XGBoost", metric)] = {
            "method": "RNA XGBoost",
            "metric": metric,
            "n_donors": "8",
            "mean": mean,
            "ci95_margin": "0.01",
            "worst_donor_value": "0.4",
            "best_donor_value": "0.9",
        }
    rows = build_rows(summary)
    assert rows[0]["method"] == "RNA XGBoost"
    assert rows[0]["accuracy"] == "0.8000 $\\pm$ 0.0100"
    assert rows[0]["modality"] == "RNA-only"


def test_build_rows_uses_full_training_protocol_caveat() -> None:
    summary = {}
    for metric, mean in [
        ("accuracy", "0.4"),
        ("weighted_f1", "0.5"),
        ("macro_f1", "0.3"),
        ("balanced_accuracy", "0.35"),
    ]:
        summary[("Early-fusion ridge", metric)] = {
            "method": "Early-fusion ridge",
            "metric": metric,
            "n_donors": "8",
            "mean": mean,
            "ci95_margin": "0.02",
            "worst_donor_value": "0.3",
            "best_donor_value": "0.6",
        }
    rows = build_rows(summary)
    assert rows[0]["method"] == "Early-fusion ridge"
    assert rows[0]["modality"] == "RNA+ADT"
    assert "full training donors" in rows[0]["protocol_caveat"]
    assert "train-capped" not in rows[0]["protocol_caveat"]


def test_method_order_excludes_capped_r_baselines() -> None:
    assert "scmap" not in METHOD_ORDER
    assert "scPred" not in METHOD_ORDER


def test_method_order_excludes_gaussian_nb_from_display_table() -> None:
    assert "Early-fusion GaussianNB" not in METHOD_ORDER
    assert "RNA GaussianNB" not in METHOD_ORDER
    assert "ADT GaussianNB" not in METHOD_ORDER


def test_build_rows_includes_celltypist_published_caveat() -> None:
    summary = {}
    for metric, mean in [
        ("accuracy", "0.86"),
        ("weighted_f1", "0.85"),
        ("macro_f1", "0.64"),
        ("balanced_accuracy", "0.60"),
    ]:
        summary[("CellTypist", metric)] = {
            "method": "CellTypist",
            "metric": metric,
            "n_donors": "8",
            "mean": mean,
            "ci95_margin": "0.01",
            "worst_donor_value": "0.83",
            "best_donor_value": "0.88",
        }
    rows = build_rows(summary)
    assert rows[0]["method"] == "CellTypist"
    assert rows[0]["modality"] == "RNA-only"
    assert "published CellTypist" in rows[0]["protocol_caveat"]
