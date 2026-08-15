from pathlib import Path

import pandas as pd

from experiments.reproducibility.v33_release_smoke_test import run_smoke


def test_smoke_fixture_uses_disjoint_donors_and_reaches_expected_accuracy(
    tmp_path: Path,
) -> None:
    fixture = tmp_path / "synthetic.csv"
    pd.DataFrame(
        [
            {
                "cell_id": "train_t",
                "donor": "D1",
                "split": "train",
                "label": "T",
                "rna_a": 4.0,
                "rna_b": 0.0,
                "adt_a": 3.0,
            },
            {
                "cell_id": "train_nk",
                "donor": "D1",
                "split": "train",
                "label": "NK",
                "rna_a": 0.0,
                "rna_b": 4.0,
                "adt_a": 0.0,
            },
            {
                "cell_id": "train_t_2",
                "donor": "D2",
                "split": "train",
                "label": "T",
                "rna_a": 3.5,
                "rna_b": 0.2,
                "adt_a": 2.8,
            },
            {
                "cell_id": "train_nk_2",
                "donor": "D2",
                "split": "train",
                "label": "NK",
                "rna_a": 0.2,
                "rna_b": 3.5,
                "adt_a": 0.1,
            },
            {
                "cell_id": "val_t",
                "donor": "D3",
                "split": "val",
                "label": "T",
                "rna_a": 3.8,
                "rna_b": 0.1,
                "adt_a": 3.1,
            },
            {
                "cell_id": "val_nk",
                "donor": "D3",
                "split": "val",
                "label": "NK",
                "rna_a": 0.1,
                "rna_b": 3.8,
                "adt_a": 0.0,
            },
            {
                "cell_id": "test_t",
                "donor": "D4",
                "split": "test",
                "label": "T",
                "rna_a": 4.2,
                "rna_b": 0.0,
                "adt_a": 3.2,
            },
            {
                "cell_id": "test_nk",
                "donor": "D4",
                "split": "test",
                "label": "NK",
                "rna_a": 0.0,
                "rna_b": 4.2,
                "adt_a": 0.0,
            },
        ]
    ).to_csv(fixture, index=False)

    result = run_smoke(fixture)

    assert result["train_donors"] == ["D1", "D2"]
    assert result["validation_accuracy"] == 1.0
    assert result["test_accuracy"] == 1.0
    assert result["test_macro_f1"] == 1.0
