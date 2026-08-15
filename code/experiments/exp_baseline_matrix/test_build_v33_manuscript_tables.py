import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest


LOCAL_DIR = Path(__file__).resolve().parent
if str(LOCAL_DIR) not in sys.path:
    sys.path.insert(0, str(LOCAL_DIR))

from build_v33_manuscript_tables import (  # noqa: E402
    DATE,
    OUTPUT_FILENAMES,
    SOURCE_FILES,
    generate_manuscript_tables,
    load_v33_sources,
)
from plot_v33_evidence_figure import (  # noqa: E402
    FIGURE_BASENAME,
    build_figure,
    generate_evidence_figure,
)


DONORS = [f"P{index}" for index in range(1, 9)]
SEEDS = [41, 42, 43]
METHODS = [
    "mlp",
    "mosaic_full",
    "mosaic_no_hsr",
    "mosaic_no_kd",
    "inference::rna_branch",
    "inference::adt_branch",
    "inference::fusion_branch",
    "inference::uniform_fusion",
    "inference::margin_gate",
    "inference::margin_gate_hsr",
]
METRICS = ["accuracy", "weighted_f1", "macro_f1", "balanced_accuracy"]
SCENARIOS = [
    ("full", 0.0),
    ("random_10", 0.1),
    ("random_30", 0.3),
    ("random_50", 0.5),
    ("random_70", 0.7),
    ("marker_memory", np.nan),
    ("marker_tcell", np.nan),
    ("rna_only", 1.0),
]
TARGETS = ["gdT_2", "NK_3", "CD4 TCM_1", "CD8 TEM_4", "B naive lambda"]
SCORES = ["one_minus_max_probability", "one_minus_margin", "energy"]
FOCUS_CLASSES = [
    "CD4 TEM_3",
    "CD4 TEM_4",
    "CD4 TCM_3",
    "CD8 Naive",
    "CD8 Naive_2",
    "CD8 TCM_1",
]
PDC_CLASSES = [
    "cd4_cm",
    "cd4_em",
    "cd4_n",
    "cd8_cm",
    "cd8_em",
    "cd8_emra",
    "cd8_n",
    "monocyte",
]


def _write_csv(root: Path, relative: str, frame: pd.DataFrame) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _write_json(root: Path, relative: str, payload: dict) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _donor_summary() -> pd.DataFrame:
    rows = []
    for method_index, method in enumerate(METHODS):
        for metric_index, metric in enumerate(METRICS):
            mean = 0.90 - method_index * 0.006 - metric_index * 0.01
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "n_donors": 8,
                    "mean": mean,
                    "sd_across_donors": 0.02,
                    "ci95_low": mean - 0.0167,
                    "ci95_high": mean + 0.0167,
                    "ci95_margin": 0.0167,
                    "worst_donor_value": mean - 0.04,
                    "best_donor_value": mean + 0.03,
                }
            )
    return pd.DataFrame(rows)


def _paired_statistics() -> pd.DataFrame:
    rows = []
    comparisons = [
        ("mosaic_full", comparator)
        for comparator in ["mlp", "mosaic_no_hsr", "mosaic_no_kd"]
    ] + [
        ("inference::margin_gate_hsr", comparator)
        for comparator in [
            "inference::rna_branch",
            "inference::adt_branch",
            "inference::fusion_branch",
            "inference::uniform_fusion",
            "inference::margin_gate",
        ]
    ]
    for reference, comparator in comparisons:
        for metric in METRICS:
            rows.append(
                {
                    "reference": reference,
                    "comparator": comparator,
                    "metric": metric,
                    "n_donors": 8,
                    "mean_difference": 0.012,
                    "difference_ci95_low": 0.002,
                    "difference_ci95_high": 0.022,
                    "difference_ci95_margin": 0.010,
                    "paired_t_pvalue": 0.02,
                    "wilcoxon_pvalue": 0.03125,
                    "positive_donor_count": 7,
                    "negative_donor_count": 1,
                    "zero_donor_count": 0,
                }
            )
    return pd.DataFrame(rows)


def _panel_metrics() -> pd.DataFrame:
    rows = []
    for donor_index, donor in enumerate(DONORS):
        for seed_index, seed in enumerate(SEEDS):
            for scenario, fraction in SCENARIOS:
                degradation = {
                    "full": 0.00,
                    "random_10": 0.01,
                    "random_30": 0.04,
                    "random_50": 0.09,
                    "random_70": 0.17,
                    "marker_memory": 0.12,
                    "marker_tcell": 0.10,
                    "rna_only": 0.25,
                }[scenario]
                base = 0.91 - 0.001 * donor_index + 0.0005 * seed_index
                rows.append(
                    {
                        "test_donor": donor,
                        "seed": seed,
                        "scenario": scenario,
                        "mask_fraction": fraction,
                        "n_masked_proteins": int(224 * (fraction or 0))
                        if not pd.isna(fraction)
                        else 8,
                        "n_test": 1000,
                        "accuracy": base - degradation,
                        "weighted_f1": base - 0.01 - degradation,
                        "macro_f1": base - 0.04 - degradation,
                        "ece": 0.04 + degradation / 2,
                        "brier": 0.08 + degradation,
                        "delta_accuracy": -degradation,
                        "delta_weighted_f1": -degradation,
                        "delta_macro_f1": -degradation,
                        "delta_ece": degradation / 2,
                        "delta_brier": degradation,
                    }
                )
    return pd.DataFrame(rows)


def _panel_slopes() -> pd.DataFrame:
    rows = []
    for donor in DONORS:
        for seed in SEEDS:
            for metric, slope in [
                ("accuracy", -0.22),
                ("weighted_f1", -0.23),
                ("macro_f1", -0.27),
                ("ece", 0.09),
                ("brier", 0.17),
            ]:
                rows.append(
                    {
                        "test_donor": donor,
                        "seed": seed,
                        "metric": metric,
                        "slope_per_full_missing_fraction": slope,
                    }
                )
    return pd.DataFrame(rows)


def _unknown_metrics() -> pd.DataFrame:
    rows = []
    for target_index, target in enumerate(TARGETS):
        for seed_index, seed in enumerate(SEEDS):
            for score_index, score in enumerate(SCORES):
                for coverage in [0.95, 0.80]:
                    recall = (
                        0.45
                        + 0.04 * target_index
                        + 0.01 * seed_index
                        + 0.02 * score_index
                        + (0.12 if coverage == 0.80 else 0.0)
                    )
                    rows.append(
                        {
                            "target_label": target,
                            "seed": seed,
                            "score": score,
                            "known_coverage_target": coverage,
                            "threshold": 0.25,
                            "known_test_coverage": coverage - 0.005,
                            "known_false_reject_rate": 1.0 - coverage + 0.005,
                            "unknown_recall": recall,
                            "unknown_auroc": 0.78 + 0.01 * score_index,
                            "unknown_auprc": 0.66 + 0.01 * score_index,
                            "known_accepted_l3_accuracy": 0.94,
                            "known_accepted_l3_risk": 0.06,
                            "known_rejected_parent_accuracy": 0.84,
                            "unknown_rejected_parent_accuracy": 0.81,
                            "unknown_parent_safe_rate": recall * 0.81,
                            "unknown_unsafe_accept_rate": 1.0 - recall,
                            "known_hierarchical_accuracy": 0.93,
                            "unknown_hierarchical_accuracy": recall * 0.81,
                            "combined_hierarchical_accuracy": 0.90,
                            "n_val_known": 800,
                            "n_test_known": 1000,
                            "n_test_unknown": 100,
                        }
                    )
    return pd.DataFrame(rows)


def _pdc_comparison() -> pd.DataFrame:
    rows = []
    for method, offset, mode in [
        ("MOSAIC-N", 0.0, "seed Student-t"),
        ("MMoCHi", -0.01, "cell bootstrap; conditional on one workflow"),
    ]:
        for metric, base in [
            ("accuracy", 0.94),
            ("weighted_f1", 0.935),
            ("macro_f1", 0.93),
        ]:
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "n_seeds": 3 if method == "MOSAIC-N" else 1,
                    "mean": base + offset,
                    "ci95_margin": 0.008,
                    "uncertainty_mode": mode,
                    "dataset": "PDC101 sorted external holdout",
                    "n_holdout": 2098,
                    "protocol_caveat": (
                        "train-only preprocessing; no HTO/control; three model seeds"
                        if method == "MOSAIC-N"
                        else "official hierarchy; external_holdout=True; one workflow fit"
                    ),
                }
            )
    return pd.DataFrame(rows)


def _pdc_per_class() -> pd.DataFrame:
    rows = []
    for method, offset, mode in [
        ("MOSAIC-N", 0.0, "seed Student-t"),
        ("MMoCHi", 0.02, "cell bootstrap; conditional on one workflow"),
    ]:
        for class_index, class_label in enumerate(PDC_CLASSES):
            rows.append(
                {
                    "method": method,
                    "class_label": class_label,
                    "n_seeds": 3 if method == "MOSAIC-N" else 1,
                    "mean_f1": 0.82 + 0.02 * class_index + offset,
                    "ci95_margin": 0.03,
                    "support": 200 + class_index,
                    "uncertainty_mode": mode,
                    "dataset": "PDC101 sorted external holdout",
                    "n_holdout": 2098,
                    "protocol_caveat": (
                        "train-only preprocessing; no HTO/control; three model seeds"
                        if method == "MOSAIC-N"
                        else "official hierarchy; external_holdout=True; one workflow fit"
                    ),
                }
            )
    return pd.DataFrame(rows)


def _attribution_stability() -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        for class_label in FOCUS_CLASSES:
            for modality, offset in [("RNA", 0.0), ("ADT", 0.08)]:
                rows.append(
                    {
                        "seed": seed,
                        "class_label": class_label,
                        "modality": modality,
                        "donor_a": "P1",
                        "donor_b": "P2",
                        "n_shared_features": 100,
                        "spearman": 0.62 + offset,
                        "top_k": 20,
                        "top_k_jaccard": 0.31 + offset,
                    }
                )
    return pd.DataFrame(rows)


def _marker_enrichment() -> pd.DataFrame:
    rows = []
    for class_label in FOCUS_CLASSES:
        for modality, offset in [("RNA", 0.0), ("ADT", 0.1)]:
            rows.append(
                {
                    "class_label": class_label,
                    "modality": modality,
                    "top_k": 20,
                    "top_features": "A;B;C",
                    "canonical_markers": "A;D",
                    "available_canonical_count": 2,
                    "observed_overlap": 1,
                    "observed_fraction": 0.5 + offset,
                    "permutation_pvalue": 0.02,
                }
            )
    return pd.DataFrame(rows)


def write_complete_v33_sources(root: Path) -> None:
    frames = {
        "donor_summary": _donor_summary(),
        "paired_statistics": _paired_statistics(),
        "panel_metrics": _panel_metrics(),
        "panel_slopes": _panel_slopes(),
        "unknown_metrics": _unknown_metrics(),
        "pdc_comparison": _pdc_comparison(),
        "pdc_per_class": _pdc_per_class(),
        "attribution_stability": _attribution_stability(),
        "marker_enrichment": _marker_enrichment(),
    }
    for key, frame in frames.items():
        _write_csv(root, SOURCE_FILES[key], frame)
    _write_json(
        root,
        SOURCE_FILES["panel_config"],
        {
            "date": DATE,
            "donors": DONORS,
            "seeds": SEEDS,
            "random_fractions": [0.1, 0.3, 0.5, 0.7],
            "mask_policy": "feature-level deterministic masks; no test-label selection",
            "missing_checkpoints": [],
        },
    )
    _write_json(
        root,
        SOURCE_FILES["unknown_config"],
        {
            "date": DATE,
            "targets": TARGETS,
            "seeds": SEEDS,
            "scores": SCORES,
            "known_coverage_targets": [0.95, 0.80],
            "threshold_source": "known validation only",
            "missing_runs": [],
        },
    )
    _write_json(
        root,
        SOURCE_FILES["pdc_config"],
        {
            "date": DATE,
            "seeds": SEEDS,
            "input_exclusions": "HTO and isotype/control proteins",
            "test_label_tuning": False,
        },
    )
    _write_json(
        root,
        SOURCE_FILES["attribution_config"],
        {
            "date": DATE,
            "focus_classes": FOCUS_CLASSES,
            "attribution": "gradient-times-input",
            "marker_enrichment_permutations": 5000,
            "missing_checkpoints": [],
        },
    )


def test_missing_v33_source_fails_without_legacy_fallback(tmp_path: Path) -> None:
    legacy = tmp_path / "results/tables/v30_main_wide_performance_table_ci_2026-07-23.csv"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("method,accuracy\nlegacy,0.99\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="V33 critical inputs are missing"):
        load_v33_sources(tmp_path)


def test_build_tables_uses_donor_unit_and_preserves_provenance(
    tmp_path: Path,
) -> None:
    write_complete_v33_sources(tmp_path)
    outputs = generate_manuscript_tables(tmp_path)

    donor = pd.read_csv(outputs["donor_performance"])
    full_accuracy = donor[
        donor["method"].eq("mosaic_full") & donor["metric"].eq("accuracy")
    ].iloc[0]
    assert full_accuracy["n_donors"] == 8
    assert full_accuracy["statistical_unit"] == "held-out donor after seed averaging"
    assert full_accuracy["mean"] == pytest.approx(0.894)
    assert full_accuracy["protocol"] == "PBMC nested leave-one-donor-out"
    assert full_accuracy["caveat"]
    assert full_accuracy["source_artifact"] == SOURCE_FILES["donor_summary"]

    mirror = tmp_path / "output/tables" / outputs["donor_performance"].name
    assert mirror.read_bytes() == outputs["donor_performance"].read_bytes()

    manifest = pd.read_csv(outputs["manifest_csv"])
    assert set(SOURCE_FILES.values()).issubset(set(manifest["source_artifact"]))
    assert manifest["sha256"].str.fullmatch(r"[0-9a-f]{64}").all()


def test_incomplete_v33_protocol_config_is_rejected(tmp_path: Path) -> None:
    write_complete_v33_sources(tmp_path)
    config_path = tmp_path / SOURCE_FILES["unknown_config"]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["missing_runs"] = ["gdT_2/mosaic_full_seed43/model.pt"]
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown_config.*missing_runs"):
        generate_manuscript_tables(tmp_path)


def test_unknown_table_averages_seeds_before_targets(tmp_path: Path) -> None:
    write_complete_v33_sources(tmp_path)
    outputs = generate_manuscript_tables(tmp_path)

    unknown = pd.read_csv(outputs["unknown_reject"])
    row = unknown[
        unknown["score"].eq("one_minus_max_probability")
        & np.isclose(unknown["known_coverage_target"], 0.95)
    ].iloc[0]
    expected_recall = np.mean([0.45 + 0.04 * index + 0.01 for index in range(5)])
    assert row["mean_unknown_recall"] == pytest.approx(expected_recall)
    assert row["n_targets"] == 5
    assert row["n_seeds"] == 3
    assert row["statistical_unit"] == "leave-class-out target after seed averaging"
    assert row["threshold_source"] == "known validation only"


def test_latex_bundle_contains_caveats_and_no_missing_values(tmp_path: Path) -> None:
    write_complete_v33_sources(tmp_path)
    outputs = generate_manuscript_tables(tmp_path)

    latex = outputs["latex_bundle"].read_text(encoding="utf-8")
    assert "held-out donor" in latex
    assert "validation-only thresholds" in latex
    assert "external\\_holdout=True" in latex
    assert "source artifacts" in latex
    assert "NaN" not in latex
    assert re.search(r"(^|&)\s*NA\s*(&|\\\\)", latex, flags=re.MULTILINE) is None


def test_latex_fragments_are_section_specific_and_horizontal(
    tmp_path: Path,
) -> None:
    write_complete_v33_sources(tmp_path)
    outputs = generate_manuscript_tables(tmp_path)

    main = outputs["main_donor_fragment"].read_text(encoding="utf-8")
    assert "Metric & MLP early fusion & MOSAIC-N without KD" in main
    assert "MOSAIC-N without HSR & MOSAIC-N full" in main
    assert main.count(r"\\") == 4
    assert "accuracy" in main
    assert "weighted\\_f1" in main
    assert "macro\\_f1" in main
    assert "balanced\\_accuracy" not in main

    for key, marker in [
        ("supplement_donor_fragment", "Worst donor"),
        ("supplement_panel_fragment", "random\\_50"),
        ("supplement_unknown_fragment", "one\\_minus\\_margin"),
        ("supplement_unknown_target_fragment", "B naive lambda"),
        ("supplement_pdc_fragment", "MMoCHi"),
        ("supplement_pdc_per_class_fragment", "cd8\\_emra"),
        ("supplement_attribution_fragment", "Permutation $p$"),
    ]:
        fragment = outputs[key].read_text(encoding="utf-8")
        assert marker in fragment
        assert r"\begin{table" not in fragment
        assert r"\end{table" not in fragment
        assert "NaN" not in fragment


def test_figure_has_six_dense_evidence_panels_and_writes_mirrors(
    tmp_path: Path,
) -> None:
    write_complete_v33_sources(tmp_path)
    generate_manuscript_tables(tmp_path)

    figure, figure_data = build_figure(tmp_path)
    try:
        assert len(figure.axes) == 6
        assert set(figure_data["panel"]) == {"a", "b", "c", "d", "e", "f"}
        assert figure_data["protocol"].str.len().gt(0).all()
        assert figure_data["caveat"].str.len().gt(0).all()
        assert figure_data["source_artifact"].str.contains("v33", case=False).all()
        reject = figure_data[figure_data["panel"].eq("d")]
        assert reject["target_label"].nunique() == 5
        assert reject["score"].nunique() == 3
        pdc = figure_data[figure_data["panel"].eq("e")]
        assert set(pdc["record_type"].dropna()) == {"aggregate", "per_class"}
        assert pdc.loc[pdc["record_type"].eq("per_class"), "class_label"].nunique() == 8
        all_text = [
            text
            for axis in figure.axes
            for text in axis.texts
        ]
        assert not any(
            re.search(r"\d+\.0/8", text.get_text())
            for text in all_text
        )
        attribution_text = {
            text.get_text(): text.get_color()
            for text in figure.axes[5].texts
        }
        assert attribution_text["0.31"] == "white"
        assert attribution_text["0.62"] != "white"
    finally:
        plt.close(figure)

    outputs = generate_evidence_figure(tmp_path, dpi=120)
    for suffix in [".pdf", ".svg", ".png"]:
        result_path = tmp_path / "results/figures" / f"{FIGURE_BASENAME}_{DATE}{suffix}"
        mirror_path = tmp_path / "output/figures" / result_path.name
        assert result_path.exists()
        assert mirror_path.read_bytes() == result_path.read_bytes()
    source_manifest = pd.read_csv(outputs["source_manifest_csv"])
    assert set(source_manifest["panel"]) == {"a", "b", "c", "d", "e", "f"}
    assert outputs["source_manifest_json"].exists()


def test_output_filenames_are_v33_only() -> None:
    assert OUTPUT_FILENAMES
    assert all("v33" in filename.lower() for filename in OUTPUT_FILENAMES.values())
    assert "v33" in FIGURE_BASENAME.lower()
