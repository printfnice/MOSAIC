import numpy as np
import pandas as pd

from analyze_v33_donor_attribution import (
    canonical_marker_enrichment,
    pairwise_feature_stability,
    pairwise_seed_stability,
    top_k_jaccard,
)


def test_top_k_jaccard_uses_feature_identity() -> None:
    assert np.isclose(top_k_jaccard(["A", "B", "C"], ["B", "C", "D"]), 0.5)
    assert np.isclose(top_k_jaccard([], []), 1.0)


def test_pairwise_feature_stability_compares_distinct_donors() -> None:
    frame = pd.DataFrame(
        {
            "test_donor": ["P1"] * 3 + ["P2"] * 3,
            "seed": [42] * 6,
            "class_label": ["X"] * 6,
            "modality": ["ADT"] * 6,
            "feature": ["A", "B", "C", "A", "B", "C"],
            "mean_abs_attribution": [3.0, 2.0, 1.0, 2.9, 2.1, 1.0],
        }
    )
    result = pairwise_feature_stability(frame, top_k=2)
    assert len(result) == 1
    assert result.iloc[0]["donor_a"] == "P1"
    assert result.iloc[0]["donor_b"] == "P2"
    assert result.iloc[0]["spearman"] > 0.9
    assert np.isclose(result.iloc[0]["top_k_jaccard"], 1.0)


def test_canonical_marker_enrichment_is_deterministic() -> None:
    result_a = canonical_marker_enrichment(
        top_features=["CD3", "CD4", "CD45RO"],
        available_features=["CD3", "CD4", "CD45RO", "CD19", "CD14", "CD56"],
        canonical_markers=["CD3", "CD4", "CD45RO"],
        n_permutations=500,
        seed=42,
    )
    result_b = canonical_marker_enrichment(
        top_features=["CD3", "CD4", "CD45RO"],
        available_features=["CD3", "CD4", "CD45RO", "CD19", "CD14", "CD56"],
        canonical_markers=["CD3", "CD4", "CD45RO"],
        n_permutations=500,
        seed=42,
    )
    assert result_a == result_b
    assert result_a["observed_hits"] == 3
    assert result_a["permutation_pvalue"] < 0.2


def test_canonical_marker_enrichment_handles_unavailable_markers() -> None:
    result = canonical_marker_enrichment(
        top_features=["A", "B"],
        available_features=["A", "B", "C"],
        canonical_markers=["CD4", "CD8"],
        n_permutations=100,
        seed=42,
    )
    assert result["observed_hits"] == 0
    assert result["available_canonical_markers"] == 0
    assert result["permutation_pvalue"] == 1.0


def test_pairwise_seed_stability_preserves_donor_identity() -> None:
    frame = pd.DataFrame(
        {
            "test_donor": ["P8"] * 6,
            "seed": [41] * 3 + [42] * 3,
            "class_label": ["X"] * 6,
            "modality": ["RNA"] * 6,
            "feature": ["A", "B", "C", "A", "B", "C"],
            "mean_abs_attribution": [3.0, 2.0, 1.0, 2.8, 2.2, 1.0],
        }
    )
    result = pairwise_seed_stability(frame, top_k=2)
    assert len(result) == 1
    assert result.iloc[0]["test_donor"] == "P8"
    assert result.iloc[0]["seed_a"] == 41
    assert result.iloc[0]["seed_b"] == 42
    assert result.iloc[0]["spearman"] > 0.9
