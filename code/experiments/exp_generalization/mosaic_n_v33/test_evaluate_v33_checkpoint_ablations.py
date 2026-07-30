import numpy as np
import torch

from evaluate_v33_checkpoint_ablations import (
    VARIANTS,
    known_label_mask,
    select_variant_logits,
    summarize_variant_predictions,
)


def test_select_variant_logits_matches_declared_architecture_outputs() -> None:
    outputs = {
        "rna_logits": torch.tensor([[1.0, 0.0]]),
        "adt_logits": torch.tensor([[0.0, 1.0]]),
        "fusion_logits": torch.tensor([[2.0, 0.0]]),
        "base_final_logits": torch.tensor([[0.5, 0.6]]),
        "final_logits": torch.tensor([[0.4, 0.8]]),
    }
    assert VARIANTS == [
        "rna_branch",
        "adt_branch",
        "fusion_branch",
        "uniform_fusion",
        "margin_gate",
        "margin_gate_hsr",
    ]
    assert torch.equal(select_variant_logits(outputs, "rna_branch"), outputs["rna_logits"])
    assert torch.equal(select_variant_logits(outputs, "margin_gate"), outputs["base_final_logits"])
    assert torch.equal(select_variant_logits(outputs, "margin_gate_hsr"), outputs["final_logits"])
    expected_uniform = (
        outputs["rna_logits"] + outputs["adt_logits"] + outputs["fusion_logits"]
    ) / 3.0
    assert torch.allclose(select_variant_logits(outputs, "uniform_fusion"), expected_uniform)


def test_known_label_mask_excludes_train_absent_classes() -> None:
    y_true = np.array([0, 1, 2, 2])
    train_labels = np.array([0, 1, 1])
    assert known_label_mask(y_true, train_labels).tolist() == [True, True, False, False]


def test_summarize_variant_predictions_reports_all_primary_metrics() -> None:
    y_true = np.array([0, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 2])
    summary = summarize_variant_predictions(y_true, y_pred)

    assert np.isclose(summary["accuracy"], 0.75)
    assert set(summary) == {
        "accuracy",
        "weighted_f1",
        "macro_f1",
        "balanced_accuracy",
    }

