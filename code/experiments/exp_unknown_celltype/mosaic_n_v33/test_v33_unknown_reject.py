import numpy as np

from build_v33_unknown_protocol import (
    TARGETS,
    build_known_training_arrays,
    validation_threshold,
)
from evaluate_v33_unknown_reject import (
    ROOT,
    evaluate_score_policy,
    evaluation_output_dir,
    hierarchical_policy_metrics,
)


def test_targets_match_the_predeclared_five_target_protocol() -> None:
    assert TARGETS == [
        "gdT_2",
        "NK_3",
        "CD4 TCM_1",
        "CD8 TEM_4",
        "B naive lambda",
    ]


def test_known_training_arrays_exclude_unknown_cells_from_training_and_test() -> None:
    source = {
        "gene": np.zeros((5, 2), dtype=np.float32),
        "protein": np.zeros((5, 1), dtype=np.float32),
        "labels": np.array([0, 1, 0, 1, -1]),
        "train_idx": np.array([0, 1]),
        "val_idx": np.array([2]),
        "test_known_idx": np.array([3]),
        "test_unknown_idx": np.array([4]),
        "cell_ids": np.array(["a", "b", "c", "d", "e"]),
        "gene_names": np.array(["g1", "g2"]),
        "protein_names": np.array(["p1"]),
        "label_classes": np.array(["A", "B"]),
    }
    arrays = build_known_training_arrays(source)
    assert arrays["test_idx"].tolist() == [3]
    assert 4 not in arrays["train_idx"]
    assert 4 not in arrays["val_idx"]
    assert 4 not in arrays["test_idx"]
    assert arrays["label_encoder"].classes_.tolist() == ["A", "B"]


def test_validation_threshold_uses_known_validation_only() -> None:
    scores = np.array([0.1, 0.2, 0.3, 0.4])
    threshold = validation_threshold(scores, known_coverage=0.75)
    assert np.isclose(threshold, 0.3)


def test_unknown_score_policy_reports_known_coverage_and_unknown_recall() -> None:
    result = evaluate_score_policy(
        validation_known_scores=np.array([0.1, 0.2, 0.3, 0.4]),
        test_known_scores=np.array([0.1, 0.2, 0.8]),
        test_unknown_scores=np.array([0.7, 0.9]),
        known_coverage=0.75,
    )
    assert np.isclose(result["threshold"], 0.3)
    assert np.isclose(result["known_test_coverage"], 2 / 3)
    assert np.isclose(result["unknown_recall"], 1.0)


def test_hierarchical_policy_uses_l3_for_accepted_and_parent_for_rejected() -> None:
    result = hierarchical_policy_metrics(
        known_l3_correct=np.array([True, False, False]),
        known_rejected=np.array([False, False, True]),
        known_parent_correct=np.array([True, True, True]),
        unknown_rejected=np.array([True, False]),
        unknown_parent_correct=np.array([True, True]),
    )
    assert np.isclose(result["known_hierarchical_accuracy"], 2 / 3)
    assert np.isclose(result["unknown_hierarchical_accuracy"], 1 / 2)
    assert np.isclose(result["combined_hierarchical_accuracy"], 3 / 5)
    assert np.isclose(result["unknown_unsafe_accept_rate"], 1 / 2)


def test_hierarchical_policy_rejects_mismatched_array_lengths() -> None:
    with np.testing.assert_raises(ValueError):
        hierarchical_policy_metrics(
            known_l3_correct=np.array([True]),
            known_rejected=np.array([False, True]),
            known_parent_correct=np.array([True]),
            unknown_rejected=np.array([True]),
            unknown_parent_correct=np.array([True]),
        )


def test_smoke_evaluation_isolated_from_formal_output(tmp_path) -> None:
    formal = ROOT / "results/exp_unknown_celltype/mosaic_n_v33"
    formal_output, is_formal = evaluation_output_dir(formal)
    assert is_formal
    assert formal_output == formal / "evaluation"

    smoke = tmp_path / "mosaic_n_v33_smoke"
    smoke_output, is_formal = evaluation_output_dir(smoke)
    assert not is_formal
    assert smoke_output == smoke / "evaluation"
