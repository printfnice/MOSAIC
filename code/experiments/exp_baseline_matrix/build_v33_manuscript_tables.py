#!/usr/bin/env python
"""Build manuscript-ready V33 evidence tables from V33 artifacts only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Tuple

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
DATE = "2026-07-23"
DONORS = [f"P{index}" for index in range(1, 9)]
SEEDS = [41, 42, 43]
METRICS = ["accuracy", "weighted_f1", "macro_f1", "balanced_accuracy"]
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
SCENARIOS = [
    "full",
    "random_10",
    "random_30",
    "random_50",
    "random_70",
    "marker_memory",
    "marker_tcell",
    "rna_only",
]
UNKNOWN_TARGETS = [
    "gdT_2",
    "NK_3",
    "CD4 TCM_1",
    "CD8 TEM_4",
    "B naive lambda",
]
UNKNOWN_SCORES = [
    "one_minus_max_probability",
    "one_minus_margin",
    "energy",
]
UNKNOWN_COVERAGES = [0.95, 0.80]
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
MODALITIES = ["RNA", "ADT"]

SOURCE_FILES = {
    "donor_summary": (
        f"results/tables/mosaic_n_v33_donor_method_summary_{DATE}.csv"
    ),
    "paired_statistics": (
        f"results/tables/mosaic_n_v33_paired_donor_statistics_{DATE}.csv"
    ),
    "panel_metrics": (
        f"results/tables/mosaic_n_v33_panel_robustness_metrics_{DATE}.csv"
    ),
    "panel_slopes": (
        f"results/tables/mosaic_n_v33_panel_robustness_slopes_{DATE}.csv"
    ),
    "panel_config": "results/exp_missing_modality/mosaic_n_v33/config.json",
    "unknown_metrics": (
        f"results/tables/mosaic_n_v33_unknown_reject_metrics_{DATE}.csv"
    ),
    "unknown_config": (
        "results/exp_unknown_celltype/mosaic_n_v33/evaluation/config.json"
    ),
    "pdc_comparison": (
        f"results/tables/mosaic_n_v33_pdc101_mmochi_comparison_{DATE}.csv"
    ),
    "pdc_per_class": (
        f"results/tables/mosaic_n_v33_pdc101_mmochi_per_class_{DATE}.csv"
    ),
    "pdc_config": "results/exp_generalization/mosaic_n_v33/pdc101/config.json",
    "attribution_stability": (
        f"results/tables/mosaic_n_v33_donor_attribution_stability_{DATE}.csv"
    ),
    "marker_enrichment": (
        f"results/tables/mosaic_n_v33_marker_enrichment_{DATE}.csv"
    ),
    "attribution_config": "results/exp_explainability/mosaic_n_v33/config.json",
}

OUTPUT_FILENAMES = {
    "donor_performance": (
        f"mosaic_n_v33_manuscript_donor_performance_{DATE}.csv"
    ),
    "paired_statistics": (
        f"mosaic_n_v33_manuscript_paired_statistics_{DATE}.csv"
    ),
    "panel_robustness": (
        f"mosaic_n_v33_manuscript_panel_robustness_{DATE}.csv"
    ),
    "panel_slopes": f"mosaic_n_v33_manuscript_panel_slopes_{DATE}.csv",
    "unknown_reject": (
        f"mosaic_n_v33_manuscript_unknown_reject_{DATE}.csv"
    ),
    "unknown_target_summary": (
        f"mosaic_n_v33_manuscript_unknown_target_summary_{DATE}.csv"
    ),
    "pdc_comparison": (
        f"mosaic_n_v33_manuscript_pdc101_comparison_{DATE}.csv"
    ),
    "pdc_per_class": (
        f"mosaic_n_v33_manuscript_pdc101_per_class_{DATE}.csv"
    ),
    "attribution_stability": (
        f"mosaic_n_v33_manuscript_attribution_stability_{DATE}.csv"
    ),
    "marker_enrichment": (
        f"mosaic_n_v33_manuscript_marker_enrichment_{DATE}.csv"
    ),
    "manifest_csv": f"mosaic_n_v33_manuscript_evidence_manifest_{DATE}.csv",
    "manifest_json": f"mosaic_n_v33_manuscript_evidence_manifest_{DATE}.json",
    "latex_bundle": f"mosaic_n_v33_manuscript_tables_{DATE}.tex",
    "main_donor_fragment": f"mosaic_n_v33_main_donor_performance_{DATE}.tex",
    "supplement_donor_fragment": (
        f"mosaic_n_v33_supplement_donor_performance_{DATE}.tex"
    ),
    "supplement_panel_fragment": (
        f"mosaic_n_v33_supplement_panel_robustness_{DATE}.tex"
    ),
    "supplement_unknown_fragment": (
        f"mosaic_n_v33_supplement_unknown_reject_{DATE}.tex"
    ),
    "supplement_unknown_target_fragment": (
        f"mosaic_n_v33_supplement_unknown_targets_{DATE}.tex"
    ),
    "supplement_pdc_fragment": (
        f"mosaic_n_v33_supplement_pdc101_{DATE}.tex"
    ),
    "supplement_pdc_per_class_fragment": (
        f"mosaic_n_v33_supplement_pdc101_per_class_{DATE}.tex"
    ),
    "supplement_attribution_fragment": (
        f"mosaic_n_v33_supplement_attribution_{DATE}.tex"
    ),
}

METHOD_LABELS = {
    "mlp": "MLP early fusion",
    "mosaic_full": "MOSAIC-N full",
    "mosaic_no_hsr": "MOSAIC-N without HSR",
    "mosaic_no_kd": "MOSAIC-N without KD",
    "inference::rna_branch": "RNA branch",
    "inference::adt_branch": "ADT branch",
    "inference::fusion_branch": "Fusion branch",
    "inference::uniform_fusion": "Uniform fusion",
    "inference::margin_gate": "Learned margin gate",
    "inference::margin_gate_hsr": "Learned margin gate + HSR",
}


def _require_columns(frame: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = set(required).difference(frame.columns)
    if missing:
        raise ValueError(f"{name} missing required columns: {sorted(missing)}")


def _require_exact_values(
    frame: pd.DataFrame,
    column: str,
    expected: Iterable[object],
    name: str,
) -> None:
    actual = set(frame[column].tolist())
    expected_set = set(expected)
    if actual != expected_set:
        raise ValueError(
            f"{name} has incomplete {column}; "
            f"missing={sorted(expected_set - actual, key=str)}, "
            f"extra={sorted(actual - expected_set, key=str)}"
        )


def _require_unique(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    columns = list(columns)
    if frame.duplicated(columns).any():
        raise ValueError(f"{name} has duplicate rows for key {columns}")


def _require_finite(
    frame: pd.DataFrame,
    columns: Iterable[str],
    name: str,
) -> None:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains non-finite values in {column}")


def _load_json(path: Path, name: str) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} is not valid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return payload


def _validate_config(
    payload: Mapping[str, object],
    name: str,
    *,
    empty_fields: Iterable[str] = (),
) -> None:
    if str(payload.get("date")) != DATE:
        raise ValueError(f"{name} date must be {DATE}")
    for field in empty_fields:
        if field not in payload:
            raise ValueError(f"{name} missing required field {field}")
        if payload[field]:
            raise ValueError(f"{name} reports incomplete {field}: {payload[field]}")


def _validate_donor_sources(
    summary: pd.DataFrame,
    paired: pd.DataFrame,
) -> None:
    _require_columns(
        summary,
        [
            "method",
            "metric",
            "n_donors",
            "mean",
            "sd_across_donors",
            "ci95_low",
            "ci95_high",
            "ci95_margin",
            "worst_donor_value",
            "best_donor_value",
        ],
        "donor_summary",
    )
    _require_exact_values(summary, "method", METHODS, "donor_summary")
    _require_exact_values(summary, "metric", METRICS, "donor_summary")
    _require_unique(summary, ["method", "metric"], "donor_summary")
    if len(summary) != len(METHODS) * len(METRICS):
        raise ValueError("donor_summary is not a complete method-by-metric matrix")
    if not summary["n_donors"].eq(8).all():
        raise ValueError("donor_summary must use all eight held-out donors")
    _require_finite(
        summary,
        [
            "mean",
            "sd_across_donors",
            "ci95_low",
            "ci95_high",
            "ci95_margin",
            "worst_donor_value",
            "best_donor_value",
        ],
        "donor_summary",
    )

    _require_columns(
        paired,
        [
            "reference",
            "comparator",
            "metric",
            "n_donors",
            "mean_difference",
            "difference_ci95_low",
            "difference_ci95_high",
            "difference_ci95_margin",
            "paired_t_pvalue",
            "wilcoxon_pvalue",
            "positive_donor_count",
            "negative_donor_count",
            "zero_donor_count",
        ],
        "paired_statistics",
    )
    _require_unique(
        paired,
        ["reference", "comparator", "metric"],
        "paired_statistics",
    )
    expected_pairs = {
        ("mosaic_full", comparator)
        for comparator in ["mlp", "mosaic_no_hsr", "mosaic_no_kd"]
    } | {
        ("inference::margin_gate_hsr", comparator)
        for comparator in [
            "inference::rna_branch",
            "inference::adt_branch",
            "inference::fusion_branch",
            "inference::uniform_fusion",
            "inference::margin_gate",
        ]
    }
    actual_pairs = set(zip(paired["reference"], paired["comparator"]))
    if actual_pairs != expected_pairs:
        raise ValueError(
            "paired_statistics has incomplete reference/comparator pairs; "
            f"missing={sorted(expected_pairs - actual_pairs)}"
        )
    _require_exact_values(paired, "metric", METRICS, "paired_statistics")
    if len(paired) != len(expected_pairs) * len(METRICS):
        raise ValueError("paired_statistics is not a complete pair-by-metric matrix")
    if not paired["n_donors"].eq(8).all():
        raise ValueError("paired_statistics must use all eight held-out donors")
    _require_finite(
        paired,
        [
            "mean_difference",
            "difference_ci95_low",
            "difference_ci95_high",
            "difference_ci95_margin",
            "paired_t_pvalue",
            "wilcoxon_pvalue",
        ],
        "paired_statistics",
    )


def _validate_panel_sources(
    metrics: pd.DataFrame,
    slopes: pd.DataFrame,
    config: Mapping[str, object],
) -> None:
    _validate_config(
        config,
        "panel_config",
        empty_fields=["missing_checkpoints"],
    )
    if list(config.get("donors", [])) != DONORS:
        raise ValueError("panel_config donors do not match the locked V33 donor list")
    if list(config.get("seeds", [])) != SEEDS:
        raise ValueError("panel_config seeds do not match the locked V33 seed list")
    if "no test-label selection" not in str(config.get("mask_policy", "")):
        raise ValueError("panel_config does not document the locked mask policy")

    metric_columns = ["accuracy", "weighted_f1", "macro_f1", "ece", "brier"]
    _require_columns(
        metrics,
        [
            "test_donor",
            "seed",
            "scenario",
            "mask_fraction",
            "n_masked_proteins",
            "n_test",
            *metric_columns,
        ],
        "panel_metrics",
    )
    _require_exact_values(metrics, "test_donor", DONORS, "panel_metrics")
    _require_exact_values(metrics, "seed", SEEDS, "panel_metrics")
    _require_exact_values(metrics, "scenario", SCENARIOS, "panel_metrics")
    _require_unique(
        metrics,
        ["test_donor", "seed", "scenario"],
        "panel_metrics",
    )
    if len(metrics) != len(DONORS) * len(SEEDS) * len(SCENARIOS):
        raise ValueError("panel_metrics is not a complete donor-seed-scenario matrix")
    _require_finite(metrics, metric_columns, "panel_metrics")

    _require_columns(
        slopes,
        [
            "test_donor",
            "seed",
            "metric",
            "slope_per_full_missing_fraction",
        ],
        "panel_slopes",
    )
    _require_exact_values(slopes, "test_donor", DONORS, "panel_slopes")
    _require_exact_values(slopes, "seed", SEEDS, "panel_slopes")
    _require_exact_values(slopes, "metric", metric_columns, "panel_slopes")
    _require_unique(slopes, ["test_donor", "seed", "metric"], "panel_slopes")
    if len(slopes) != len(DONORS) * len(SEEDS) * len(metric_columns):
        raise ValueError("panel_slopes is not a complete donor-seed-metric matrix")
    _require_finite(
        slopes,
        ["slope_per_full_missing_fraction"],
        "panel_slopes",
    )


def _validate_unknown_sources(
    metrics: pd.DataFrame,
    config: Mapping[str, object],
) -> None:
    _validate_config(
        config,
        "unknown_config",
        empty_fields=["missing_runs"],
    )
    if list(config.get("targets", [])) != UNKNOWN_TARGETS:
        raise ValueError("unknown_config targets do not match the preregistered list")
    if list(config.get("seeds", [])) != SEEDS:
        raise ValueError("unknown_config seeds do not match the locked V33 seed list")
    if list(config.get("scores", [])) != UNKNOWN_SCORES:
        raise ValueError("unknown_config scores do not match the locked score list")
    if set(config.get("known_coverage_targets", [])) != set(UNKNOWN_COVERAGES):
        raise ValueError("unknown_config coverages do not match the locked targets")
    if str(config.get("threshold_source")) != "known validation only":
        raise ValueError("unknown_config thresholds are not validation-only")

    metric_columns = [
        "known_test_coverage",
        "unknown_recall",
        "unknown_auroc",
        "unknown_auprc",
        "known_accepted_l3_accuracy",
        "known_accepted_l3_risk",
        "known_rejected_parent_accuracy",
        "unknown_parent_safe_rate",
        "unknown_unsafe_accept_rate",
        "known_hierarchical_accuracy",
        "unknown_hierarchical_accuracy",
        "combined_hierarchical_accuracy",
    ]
    _require_columns(
        metrics,
        [
            "target_label",
            "seed",
            "score",
            "known_coverage_target",
            *metric_columns,
        ],
        "unknown_metrics",
    )
    _require_exact_values(
        metrics,
        "target_label",
        UNKNOWN_TARGETS,
        "unknown_metrics",
    )
    _require_exact_values(metrics, "seed", SEEDS, "unknown_metrics")
    _require_exact_values(metrics, "score", UNKNOWN_SCORES, "unknown_metrics")
    _require_exact_values(
        metrics,
        "known_coverage_target",
        UNKNOWN_COVERAGES,
        "unknown_metrics",
    )
    _require_unique(
        metrics,
        ["target_label", "seed", "score", "known_coverage_target"],
        "unknown_metrics",
    )
    expected_rows = (
        len(UNKNOWN_TARGETS)
        * len(SEEDS)
        * len(UNKNOWN_SCORES)
        * len(UNKNOWN_COVERAGES)
    )
    if len(metrics) != expected_rows:
        raise ValueError("unknown_metrics is not a complete target-seed-policy matrix")
    _require_finite(metrics, metric_columns, "unknown_metrics")


def _validate_pdc_sources(
    comparison: pd.DataFrame,
    per_class: pd.DataFrame,
    config: Mapping[str, object],
) -> None:
    _validate_config(config, "pdc_config")
    if list(config.get("seeds", [])) != SEEDS:
        raise ValueError("pdc_config seeds do not match the locked V33 seed list")
    if config.get("test_label_tuning") is not False:
        raise ValueError("pdc_config must explicitly disable test-label tuning")
    exclusions = str(config.get("input_exclusions", ""))
    if "HTO" not in exclusions or "control" not in exclusions:
        raise ValueError("pdc_config does not document HTO/control exclusion")

    _require_columns(
        comparison,
        [
            "method",
            "metric",
            "n_seeds",
            "mean",
            "ci95_margin",
            "uncertainty_mode",
            "dataset",
            "n_holdout",
            "protocol_caveat",
        ],
        "pdc_comparison",
    )
    _require_exact_values(
        comparison,
        "method",
        ["MOSAIC-N", "MMoCHi"],
        "pdc_comparison",
    )
    _require_exact_values(
        comparison,
        "metric",
        ["accuracy", "weighted_f1", "macro_f1"],
        "pdc_comparison",
    )
    _require_unique(comparison, ["method", "metric"], "pdc_comparison")
    if len(comparison) != 6:
        raise ValueError("pdc_comparison must contain two methods by three metrics")
    if not comparison["n_holdout"].eq(2098).all():
        raise ValueError("pdc_comparison does not use the locked 2,098-cell holdout")
    if comparison["protocol_caveat"].fillna("").str.strip().eq("").any():
        raise ValueError("pdc_comparison contains an empty protocol caveat")
    _require_finite(comparison, ["mean", "ci95_margin"], "pdc_comparison")

    _require_columns(
        per_class,
        [
            "method",
            "class_label",
            "n_seeds",
            "mean_f1",
            "ci95_margin",
            "support",
            "uncertainty_mode",
            "dataset",
            "n_holdout",
            "protocol_caveat",
        ],
        "pdc_per_class",
    )
    _require_exact_values(
        per_class,
        "method",
        ["MOSAIC-N", "MMoCHi"],
        "pdc_per_class",
    )
    _require_exact_values(
        per_class,
        "class_label",
        PDC_CLASSES,
        "pdc_per_class",
    )
    _require_unique(
        per_class,
        ["method", "class_label"],
        "pdc_per_class",
    )
    if len(per_class) != 2 * len(PDC_CLASSES):
        raise ValueError("pdc_per_class must contain two methods by eight classes")
    if not per_class["n_holdout"].eq(2098).all():
        raise ValueError("pdc_per_class does not use the locked 2,098-cell holdout")
    if per_class["protocol_caveat"].fillna("").str.strip().eq("").any():
        raise ValueError("pdc_per_class contains an empty protocol caveat")
    _require_finite(
        per_class,
        ["mean_f1", "ci95_margin", "support"],
        "pdc_per_class",
    )


def _validate_attribution_sources(
    stability: pd.DataFrame,
    enrichment: pd.DataFrame,
    config: Mapping[str, object],
) -> None:
    _validate_config(
        config,
        "attribution_config",
        empty_fields=["missing_checkpoints"],
    )
    if list(config.get("focus_classes", [])) != FOCUS_CLASSES:
        raise ValueError("attribution_config focus classes are incomplete")
    if str(config.get("attribution")) != "gradient-times-input":
        raise ValueError("attribution_config method is not gradient-times-input")

    _require_columns(
        stability,
        [
            "seed",
            "class_label",
            "modality",
            "donor_a",
            "donor_b",
            "n_shared_features",
            "spearman",
            "top_k",
            "top_k_jaccard",
        ],
        "attribution_stability",
    )
    _require_exact_values(stability, "seed", SEEDS, "attribution_stability")
    _require_exact_values(
        stability,
        "class_label",
        FOCUS_CLASSES,
        "attribution_stability",
    )
    _require_exact_values(
        stability,
        "modality",
        MODALITIES,
        "attribution_stability",
    )
    _require_finite(
        stability,
        ["spearman", "top_k_jaccard"],
        "attribution_stability",
    )
    minimum_groups = len(SEEDS) * len(FOCUS_CLASSES) * len(MODALITIES)
    observed_groups = stability[
        ["seed", "class_label", "modality"]
    ].drop_duplicates()
    if len(observed_groups) != minimum_groups:
        raise ValueError("attribution_stability is missing seed-class-modality groups")

    _require_columns(
        enrichment,
        [
            "class_label",
            "modality",
            "top_k",
            "top_features",
            "canonical_markers",
            "available_canonical_count",
            "observed_overlap",
            "observed_fraction",
            "permutation_pvalue",
        ],
        "marker_enrichment",
    )
    _require_exact_values(
        enrichment,
        "class_label",
        FOCUS_CLASSES,
        "marker_enrichment",
    )
    _require_exact_values(
        enrichment,
        "modality",
        MODALITIES,
        "marker_enrichment",
    )
    _require_unique(
        enrichment,
        ["class_label", "modality"],
        "marker_enrichment",
    )
    if len(enrichment) != len(FOCUS_CLASSES) * len(MODALITIES):
        raise ValueError("marker_enrichment is not a complete class-modality matrix")
    _require_finite(
        enrichment,
        ["observed_fraction", "permutation_pvalue"],
        "marker_enrichment",
    )


def load_v33_sources(root: Path = ROOT) -> Dict[str, object]:
    """Load and validate the complete V33 evidence contract."""

    root = Path(root)
    paths = {name: root / relative for name, relative in SOURCE_FILES.items()}
    missing = [
        str(path.relative_to(root))
        for path in paths.values()
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "V33 critical inputs are missing; no V31/V32 performance fallback is "
            f"permitted: {missing}"
        )

    sources: Dict[str, object] = {}
    for name, path in paths.items():
        if path.suffix.lower() == ".csv":
            sources[name] = pd.read_csv(path)
        elif path.suffix.lower() == ".json":
            sources[name] = _load_json(path, name)
        else:
            raise ValueError(f"unsupported V33 source type: {path}")

    _validate_donor_sources(
        sources["donor_summary"],
        sources["paired_statistics"],
    )
    _validate_panel_sources(
        sources["panel_metrics"],
        sources["panel_slopes"],
        sources["panel_config"],
    )
    _validate_unknown_sources(
        sources["unknown_metrics"],
        sources["unknown_config"],
    )
    _validate_pdc_sources(
        sources["pdc_comparison"],
        sources["pdc_per_class"],
        sources["pdc_config"],
    )
    _validate_attribution_sources(
        sources["attribution_stability"],
        sources["marker_enrichment"],
        sources["attribution_config"],
    )
    return sources


def _t_summary(values: np.ndarray) -> Tuple[float, float, float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    margin = (
        float(stats.t.ppf(0.975, len(values) - 1) * sd / np.sqrt(len(values)))
        if len(values) > 1
        else 0.0
    )
    return mean, sd, mean - margin, mean + margin


def build_donor_performance(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output.insert(1, "method_label", output["method"].map(METHOD_LABELS))
    output["statistical_unit"] = "held-out donor after seed averaging"
    output["protocol"] = "PBMC nested leave-one-donor-out"
    output["caveat"] = (
        "Eight held-out donors are the biological replicates; three model seeds "
        "are averaged within donor and are not treated as independent replicates."
    )
    output["source_artifact"] = SOURCE_FILES["donor_summary"]
    return output


def build_paired_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["reference_label"] = output["reference"].map(METHOD_LABELS)
    output["comparator_label"] = output["comparator"].map(METHOD_LABELS)
    output["statistical_unit"] = "paired held-out donor"
    output["protocol"] = "PBMC nested leave-one-donor-out"
    output["caveat"] = (
        "Paired t intervals are primary; Wilcoxon p-values are a sensitivity "
        "analysis over the same eight donors without multiplicity claims."
    )
    output["source_artifact"] = SOURCE_FILES["paired_statistics"]
    return output


def _average_seeds_within_donor(
    frame: pd.DataFrame,
    group_columns: Iterable[str],
    value_columns: Iterable[str],
) -> pd.DataFrame:
    return (
        frame.groupby([*group_columns, "test_donor"], as_index=False)[
            list(value_columns)
        ]
        .mean()
    )


def build_panel_robustness(frame: pd.DataFrame) -> pd.DataFrame:
    value_columns = ["accuracy", "weighted_f1", "macro_f1", "ece", "brier"]
    donor_means = _average_seeds_within_donor(
        frame,
        ["scenario"],
        value_columns,
    )
    scenario_metadata = (
        frame.groupby("scenario", as_index=False)
        .agg(
            mask_fraction=("mask_fraction", "mean"),
            mean_masked_proteins=("n_masked_proteins", "mean"),
        )
    )
    rows = []
    for scenario, group in donor_means.groupby("scenario", sort=False):
        for metric in value_columns:
            mean, sd, low, high = _t_summary(group[metric].to_numpy(dtype=float))
            rows.append(
                {
                    "scenario": scenario,
                    "metric": metric,
                    "n_donors": len(group),
                    "mean": mean,
                    "sd_across_donors": sd,
                    "ci95_low": low,
                    "ci95_high": high,
                    "worst_donor_value": (
                        float(group[metric].max())
                        if metric in {"ece", "brier"}
                        else float(group[metric].min())
                    ),
                }
            )
    output = pd.DataFrame(rows).merge(scenario_metadata, on="scenario", how="left")
    output["statistical_unit"] = "held-out donor after seed averaging"
    output["protocol"] = "locked feature-level protein-panel masks"
    output["caveat"] = (
        "Mask scenarios and seeds were fixed before test evaluation; marker-group "
        "rows have no scalar random-mask fraction."
    )
    output["source_artifact"] = SOURCE_FILES["panel_metrics"]
    return output


def build_panel_slopes(frame: pd.DataFrame) -> pd.DataFrame:
    donor_means = _average_seeds_within_donor(
        frame,
        ["metric"],
        ["slope_per_full_missing_fraction"],
    )
    rows = []
    for metric, group in donor_means.groupby("metric", sort=False):
        values = group["slope_per_full_missing_fraction"].to_numpy(dtype=float)
        mean, sd, low, high = _t_summary(values)
        rows.append(
            {
                "metric": metric,
                "n_donors": len(group),
                "mean_slope": mean,
                "sd_across_donors": sd,
                "ci95_low": low,
                "ci95_high": high,
                "statistical_unit": "held-out donor after seed averaging",
                "protocol": "random protein masking at 0/10/30/50/70 percent",
                "caveat": (
                    "Linear slope is a compact degradation summary over the "
                    "predeclared random-mask fractions, not a mechanistic fit."
                ),
                "source_artifact": SOURCE_FILES["panel_slopes"],
            }
        )
    return pd.DataFrame(rows)


def build_unknown_reject(frame: pd.DataFrame) -> pd.DataFrame:
    value_columns = [
        "known_test_coverage",
        "unknown_recall",
        "unknown_auroc",
        "unknown_auprc",
        "known_accepted_l3_accuracy",
        "known_accepted_l3_risk",
        "known_rejected_parent_accuracy",
        "unknown_parent_safe_rate",
        "unknown_unsafe_accept_rate",
        "known_hierarchical_accuracy",
        "unknown_hierarchical_accuracy",
        "combined_hierarchical_accuracy",
    ]
    target_means = (
        frame.groupby(
            ["target_label", "score", "known_coverage_target"],
            as_index=False,
        )[value_columns]
        .mean()
    )
    rows = []
    for (score, coverage), group in target_means.groupby(
        ["score", "known_coverage_target"],
        sort=False,
    ):
        row = {
            "score": score,
            "known_coverage_target": float(coverage),
            "n_targets": int(group["target_label"].nunique()),
            "n_seeds": len(SEEDS),
        }
        for metric in value_columns:
            values = group[metric].to_numpy(dtype=float)
            mean, sd, low, high = _t_summary(values)
            row[f"mean_{metric}"] = mean
            row[f"sd_across_targets_{metric}"] = sd
            row[f"ci95_low_{metric}"] = low
            row[f"ci95_high_{metric}"] = high
            row[f"worst_target_{metric}"] = (
                float(values.min())
                if metric not in {"known_rejected_parent_accuracy"}
                else float(values.min())
            )
        rows.append(row)
    output = pd.DataFrame(rows)
    output["statistical_unit"] = "leave-class-out target after seed averaging"
    output["threshold_source"] = "known validation only"
    output["protocol"] = "five preregistered leave-class-out targets"
    output["caveat"] = (
        "Validation-only thresholds; this is a pseudo-unknown stress test and "
        "not proof of unrestricted open-world recognition."
    )
    output["source_artifact"] = SOURCE_FILES["unknown_metrics"]
    return output


def build_unknown_target_summary(frame: pd.DataFrame) -> pd.DataFrame:
    selected = frame[
        frame["score"].eq("one_minus_margin")
        & np.isclose(frame["known_coverage_target"], 0.80)
    ].copy()
    value_columns = [
        "known_test_coverage",
        "unknown_recall",
        "unknown_auroc",
        "unknown_parent_safe_rate",
        "combined_hierarchical_accuracy",
    ]
    aggregations = {
        column: (column, "mean")
        for column in value_columns
    }
    aggregations["n_unknown"] = ("n_test_unknown", "max")
    output = (
        selected.groupby("target_label", as_index=False)
        .agg(**aggregations)
    )
    output["n_seeds"] = len(SEEDS)
    output["score"] = "one_minus_margin"
    output["known_coverage_target"] = 0.80
    output["statistical_unit"] = "leave-class-out target"
    output["protocol"] = "validation-selected margin reject at target coverage 0.80"
    output["caveat"] = (
        "Seed means are descriptive within each target; target support and "
        "difficulty vary substantially."
    )
    output["source_artifact"] = SOURCE_FILES["unknown_metrics"]
    return output


def build_pdc_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["statistical_unit"] = output["method"].map(
        {
            "MOSAIC-N": "model seed",
            "MMoCHi": "cell bootstrap conditional on one workflow fit",
        }
    )
    output["protocol"] = "PDC101 locked sorted external holdout"
    output["caveat"] = output["protocol_caveat"]
    output["source_artifact"] = SOURCE_FILES["pdc_comparison"]
    return output


def build_pdc_per_class(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["statistical_unit"] = output["method"].map(
        {
            "MOSAIC-N": "model seed",
            "MMoCHi": "cell bootstrap conditional on one workflow fit",
        }
    )
    output["protocol"] = "PDC101 locked sorted external holdout"
    output["caveat"] = output["protocol_caveat"]
    output["source_artifact"] = SOURCE_FILES["pdc_per_class"]
    return output


def build_attribution_stability(frame: pd.DataFrame) -> pd.DataFrame:
    pair_seed_means = (
        frame.groupby(
            ["class_label", "modality", "donor_a", "donor_b"],
            as_index=False,
        )[["spearman", "top_k_jaccard"]]
        .mean()
    )
    rows = []
    for (class_label, modality), group in pair_seed_means.groupby(
        ["class_label", "modality"],
        sort=False,
    ):
        rows.append(
            {
                "class_label": class_label,
                "modality": modality,
                "n_donor_pairs": len(group),
                "mean_spearman": float(group["spearman"].mean()),
                "median_spearman": float(group["spearman"].median()),
                "mean_top20_jaccard": float(group["top_k_jaccard"].mean()),
                "median_top20_jaccard": float(group["top_k_jaccard"].median()),
                "statistical_unit": "descriptive donor pair after seed averaging",
                "protocol": "donor-stratified gradient-times-input attribution",
                "caveat": (
                    "Donor pairs overlap and are not independent replicates; rank "
                    "and top-20 overlap are reported descriptively."
                ),
                "source_artifact": SOURCE_FILES["attribution_stability"],
            }
        )
    return pd.DataFrame(rows)


def build_marker_enrichment(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["statistical_unit"] = "predeclared class-modality pair"
    output["protocol"] = "canonical-marker top-20 permutation enrichment"
    output["caveat"] = (
        "Canonical marker sets and 5,000 permutations quantify enrichment; "
        "they do not establish causal regulatory mechanisms."
    )
    output["source_artifact"] = SOURCE_FILES["marker_enrichment"]
    return output


def _escape_latex(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def _format_interval(point: float, low: float, high: float) -> str:
    return f"{point:.4f} [{low:.4f}, {high:.4f}]"


def render_latex_fragments(
    tables: Mapping[str, pd.DataFrame],
) -> Dict[str, str]:
    donor = tables["donor_performance"]
    panel = tables["panel_robustness"]
    unknown = tables["unknown_reject"]
    unknown_targets = tables["unknown_target_summary"]
    pdc = tables["pdc_comparison"]
    pdc_per_class = tables["pdc_per_class"]
    attribution = tables["attribution_stability"]
    enrichment = tables["marker_enrichment"]

    main_methods = [
        "mlp",
        "mosaic_no_kd",
        "mosaic_no_hsr",
        "mosaic_full",
    ]
    main_metrics = ["accuracy", "weighted_f1", "macro_f1"]
    main = donor.set_index(["method", "metric"])
    main_lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        (
            r"Metric & MLP early fusion & MOSAIC-N without KD & "
            r"MOSAIC-N without HSR & MOSAIC-N full \\"
        ),
        r"\midrule",
    ]
    for metric in main_metrics:
        cells = []
        for method in main_methods:
            row = main.loc[(method, metric)]
            cells.append(f"{row['mean']:.4f} $\\pm$ {row['ci95_margin']:.4f}")
        main_lines.append(
            f"{_escape_latex(metric)} & " + " & ".join(cells) + r" \\"
        )
    main_lines.extend([r"\bottomrule", r"\end{tabular}"])

    donor_lines = [
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Method & Metric & Mean [95\% CI] & Worst donor & $n$ donors \\",
        r"\midrule",
    ]
    for row in donor.itertuples(index=False):
        if row.metric not in set(main_metrics):
            continue
        donor_lines.append(
            f"{_escape_latex(row.method_label)} & {_escape_latex(row.metric)} & "
            f"{_format_interval(row.mean, row.ci95_low, row.ci95_high)} & "
            f"{row.worst_donor_value:.4f} & {int(row.n_donors)}"
            + r" \\"
        )
    donor_lines.extend([r"\bottomrule", r"\end{tabular}"])

    panel_lines = [
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Scenario & Metric & Mean [95\% CI] & Worst donor & $n$ donors \\",
        r"\midrule",
    ]
    for row in panel.itertuples(index=False):
        if row.metric not in {"weighted_f1", "macro_f1", "ece"}:
            continue
        panel_lines.append(
            f"{_escape_latex(row.scenario)} & {_escape_latex(row.metric)} & "
            f"{_format_interval(row.mean, row.ci95_low, row.ci95_high)} & "
            f"{row.worst_donor_value:.4f} & {int(row.n_donors)}"
            + r" \\"
        )
    panel_lines.extend([r"\bottomrule", r"\end{tabular}"])

    unknown_lines = [
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        (
            r"Score & Target coverage & Observed coverage & Unknown recall & "
            r"AUROC & Parent-safe rate & Hierarchical accuracy \\"
        ),
        r"\midrule",
    ]
    for row in unknown.itertuples(index=False):
        unknown_lines.append(
            f"{_escape_latex(row.score)} & {row.known_coverage_target:.2f} & "
            f"{row.mean_known_test_coverage:.4f} & "
            f"{row.mean_unknown_recall:.4f} & {row.mean_unknown_auroc:.4f} & "
            f"{row.mean_unknown_parent_safe_rate:.4f} & "
            f"{row.mean_combined_hierarchical_accuracy:.4f}"
            + r" \\"
        )
    unknown_lines.extend([r"\bottomrule", r"\end{tabular}"])

    unknown_target_lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        (
            r"Target & $n$ unknown & Known coverage & Unknown recall & AUROC & "
            r"Parent-safe rate & Hierarchical accuracy \\"
        ),
        r"\midrule",
    ]
    for row in unknown_targets.itertuples(index=False):
        unknown_target_lines.append(
            f"{_escape_latex(row.target_label)} & {int(row.n_unknown)} & "
            f"{row.known_test_coverage:.4f} & {row.unknown_recall:.4f} & "
            f"{row.unknown_auroc:.4f} & {row.unknown_parent_safe_rate:.4f} & "
            f"{row.combined_hierarchical_accuracy:.4f}"
            + r" \\"
        )
    unknown_target_lines.extend([r"\bottomrule", r"\end{tabular}"])

    pdc_lines = [
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"Method & Metric & Mean $\pm$ uncertainty & Mode \\",
        r"\midrule",
    ]
    for row in pdc.itertuples(index=False):
        pdc_lines.append(
            f"{_escape_latex(row.method)} & {_escape_latex(row.metric)} & "
            f"{row.mean:.4f} $\\pm$ {row.ci95_margin:.4f} & "
            f"{_escape_latex(row.uncertainty_mode)}"
            + r" \\"
        )
    pdc_lines.extend([r"\bottomrule", r"\end{tabular}"])

    pdc_class_lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Class & Support & MOSAIC-N F1 & MMoCHi F1 \\",
        r"\midrule",
    ]
    pdc_wide = pdc_per_class.pivot(
        index=["class_label", "support"],
        columns="method",
        values="mean_f1",
    ).reset_index()
    pdc_wide["class_order"] = pdc_wide["class_label"].map(
        {label: index for index, label in enumerate(PDC_CLASSES)}
    )
    pdc_wide = pdc_wide.sort_values("class_order")
    for _, row in pdc_wide.iterrows():
        pdc_class_lines.append(
            f"{_escape_latex(row['class_label'])} & {int(row['support'])} & "
            f"{row['MOSAIC-N']:.4f} & {row['MMoCHi']:.4f}"
            + r" \\"
        )
    pdc_class_lines.extend([r"\bottomrule", r"\end{tabular}"])

    attribution_lines = [
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        (
            r"Class & Modality & Mean Spearman & Top-20 Jaccard & "
            r"Marker overlap & Permutation $p$ \\"
        ),
        r"\midrule",
    ]
    merged = attribution.merge(
        enrichment[
            [
                "class_label",
                "modality",
                "observed_fraction",
                "permutation_pvalue",
            ]
        ],
        on=["class_label", "modality"],
        validate="one_to_one",
    )
    for row in merged.itertuples(index=False):
        attribution_lines.append(
            f"{_escape_latex(row.class_label)} & {_escape_latex(row.modality)} & "
            f"{row.mean_spearman:.4f} & {row.mean_top20_jaccard:.4f} & "
            f"{row.observed_fraction:.4f} & {row.permutation_pvalue:.4g}"
            + r" \\"
        )
    attribution_lines.extend([r"\bottomrule", r"\end{tabular}"])

    return {
        "main_donor_fragment": "\n".join(main_lines) + "\n",
        "supplement_donor_fragment": "\n".join(donor_lines) + "\n",
        "supplement_panel_fragment": "\n".join(panel_lines) + "\n",
        "supplement_unknown_fragment": "\n".join(unknown_lines) + "\n",
        "supplement_unknown_target_fragment": (
            "\n".join(unknown_target_lines) + "\n"
        ),
        "supplement_pdc_fragment": "\n".join(pdc_lines) + "\n",
        "supplement_pdc_per_class_fragment": (
            "\n".join(pdc_class_lines) + "\n"
        ),
        "supplement_attribution_fragment": (
            "\n".join(attribution_lines) + "\n"
        ),
    }


def render_latex_bundle(tables: Mapping[str, pd.DataFrame]) -> str:
    donor = tables["donor_performance"]
    panel = tables["panel_robustness"]
    unknown = tables["unknown_reject"]
    pdc = tables["pdc_comparison"]
    attribution = tables["attribution_stability"]
    enrichment = tables["marker_enrichment"]

    lines = [
        "% Auto-generated from complete V33 artifacts; do not edit values manually.",
        r"\begin{table*}[t]",
        r"\centering\small",
        r"\caption{Nested donor-disjoint performance. Intervals use held-out donor as the statistical unit.}",
        r"\label{tab:v33-donor-performance}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Method & Metric & Mean [95\% CI] & Worst donor & $n$ donors \\",
        r"\midrule",
    ]
    for row in donor.itertuples(index=False):
        if row.metric not in {"accuracy", "weighted_f1", "macro_f1"}:
            continue
        lines.append(
            f"{_escape_latex(row.method_label)} & {_escape_latex(row.metric)} & "
            f"{_format_interval(row.mean, row.ci95_low, row.ci95_high)} & "
            f"{row.worst_donor_value:.4f} & {int(row.n_donors)}"
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\par\footnotesize Caveat: eight held-out donors are the biological replicates; seeds are averaged within donor.",
            rf"\par\footnotesize source artifacts: \texttt{{{_escape_latex(SOURCE_FILES['donor_summary'])}}}; "
            rf"\texttt{{{_escape_latex(SOURCE_FILES['paired_statistics'])}}}.",
            r"\end{table*}",
            "",
            r"\begin{table*}[t]",
            r"\centering\small",
            r"\caption{Missing-protein and panel-mismatch robustness under locked feature-level masks.}",
            r"\label{tab:v33-panel-robustness}",
            r"\begin{tabular}{llccc}",
            r"\toprule",
            r"Scenario & Metric & Mean [95\% CI] & Worst donor & $n$ donors \\",
            r"\midrule",
        ]
    )
    for row in panel.itertuples(index=False):
        if row.metric not in {"weighted_f1", "macro_f1", "ece"}:
            continue
        lines.append(
            f"{_escape_latex(row.scenario)} & {_escape_latex(row.metric)} & "
            f"{_format_interval(row.mean, row.ci95_low, row.ci95_high)} & "
            f"{row.worst_donor_value:.4f} & {int(row.n_donors)}"
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\par\footnotesize Caveat: masks were fixed without test-label selection; marker-group scenarios have no scalar random-mask fraction.",
            rf"\par\footnotesize source artifacts: \texttt{{{_escape_latex(SOURCE_FILES['panel_metrics'])}}}; "
            rf"\texttt{{{_escape_latex(SOURCE_FILES['panel_config'])}}}.",
            r"\end{table*}",
            "",
            r"\begin{table*}[t]",
            r"\centering\small",
            r"\caption{Leave-class-out unknown/reject stress test using validation-only thresholds.}",
            r"\label{tab:v33-unknown-reject}",
            r"\begin{tabular}{lcccccc}",
            r"\toprule",
            r"Score & Target coverage & Observed coverage & Unknown recall & AUROC & Parent-safe rate & Hierarchical accuracy \\",
            r"\midrule",
        ]
    )
    for row in unknown.itertuples(index=False):
        lines.append(
            f"{_escape_latex(row.score)} & {row.known_coverage_target:.2f} & "
            f"{row.mean_known_test_coverage:.4f} & "
            f"{row.mean_unknown_recall:.4f} & {row.mean_unknown_auroc:.4f} & "
            f"{row.mean_unknown_parent_safe_rate:.4f} & "
            f"{row.mean_combined_hierarchical_accuracy:.4f}"
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\par\footnotesize Caveat: validation-only thresholds; pseudo-unknown leave-class-out evidence does not prove unrestricted open-world recognition.",
            rf"\par\footnotesize source artifacts: \texttt{{{_escape_latex(SOURCE_FILES['unknown_metrics'])}}}; "
            rf"\texttt{{{_escape_latex(SOURCE_FILES['unknown_config'])}}}.",
            r"\end{table*}",
            "",
            r"\begin{table}[t]",
            r"\centering\small",
            r"\caption{PDC101 same-holdout comparison.}",
            r"\label{tab:v33-pdc101}",
            r"\begin{tabular}{llcc}",
            r"\toprule",
            r"Method & Metric & Mean $\pm$ uncertainty & Mode \\",
            r"\midrule",
        ]
    )
    for row in pdc.itertuples(index=False):
        lines.append(
            f"{_escape_latex(row.method)} & {_escape_latex(row.metric)} & "
            f"{row.mean:.4f} $\\pm$ {row.ci95_margin:.4f} & "
            f"{_escape_latex(row.uncertainty_mode)}"
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\par\footnotesize Caveat: MMoCHi uses the official hierarchy with external\_holdout=True and one workflow fit; uncertainty modes differ.",
            rf"\par\footnotesize source artifacts: \texttt{{{_escape_latex(SOURCE_FILES['pdc_comparison'])}}}; "
            rf"\texttt{{{_escape_latex(SOURCE_FILES['pdc_config'])}}}.",
            r"\end{table}",
            "",
            r"\begin{table*}[t]",
            r"\centering\small",
            r"\caption{Cross-donor attribution stability and canonical-marker enrichment.}",
            r"\label{tab:v33-attribution}",
            r"\begin{tabular}{llcccc}",
            r"\toprule",
            r"Class & Modality & Mean Spearman & Top-20 Jaccard & Marker overlap & Permutation $p$ \\",
            r"\midrule",
        ]
    )
    merged = attribution.merge(
        enrichment[
            [
                "class_label",
                "modality",
                "observed_fraction",
                "permutation_pvalue",
            ]
        ],
        on=["class_label", "modality"],
        validate="one_to_one",
    )
    for row in merged.itertuples(index=False):
        lines.append(
            f"{_escape_latex(row.class_label)} & {_escape_latex(row.modality)} & "
            f"{row.mean_spearman:.4f} & {row.mean_top20_jaccard:.4f} & "
            f"{row.observed_fraction:.4f} & {row.permutation_pvalue:.4g}"
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\par\footnotesize Caveat: overlapping donor pairs are descriptive; marker enrichment does not establish causality.",
            rf"\par\footnotesize source artifacts: \texttt{{{_escape_latex(SOURCE_FILES['attribution_stability'])}}}; "
            rf"\texttt{{{_escape_latex(SOURCE_FILES['marker_enrichment'])}}}.",
            r"\end{table*}",
        ]
    )
    return "\n".join(lines) + "\n"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_source_manifest(root: Path) -> pd.DataFrame:
    rows = []
    for role, relative in SOURCE_FILES.items():
        path = root / relative
        rows.append(
            {
                "evidence_role": role,
                "source_artifact": relative,
                "artifact_type": path.suffix.lower().lstrip("."),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "protocol_preserved": "yes",
                "caveat_preserved": "yes",
                "legacy_performance_fallback": "no",
            }
        )
    return pd.DataFrame(rows)


def _write_mirrored_bytes(
    root: Path,
    filename: str,
    content: bytes,
) -> Path:
    result_path = root / "results/tables" / filename
    mirror_path = root / "output/tables" / filename
    for path in [result_path, mirror_path]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return result_path


def _write_mirrored_frame(root: Path, filename: str, frame: pd.DataFrame) -> Path:
    return _write_mirrored_bytes(
        root,
        filename,
        frame.to_csv(index=False).encode("utf-8"),
    )


def generate_manuscript_tables(root: Path = ROOT) -> Dict[str, Path]:
    """Validate all V33 inputs and write mirrored manuscript table artifacts."""

    root = Path(root)
    sources = load_v33_sources(root)
    tables = {
        "donor_performance": build_donor_performance(sources["donor_summary"]),
        "paired_statistics": build_paired_statistics(
            sources["paired_statistics"]
        ),
        "panel_robustness": build_panel_robustness(sources["panel_metrics"]),
        "panel_slopes": build_panel_slopes(sources["panel_slopes"]),
        "unknown_reject": build_unknown_reject(sources["unknown_metrics"]),
        "unknown_target_summary": build_unknown_target_summary(
            sources["unknown_metrics"]
        ),
        "pdc_comparison": build_pdc_comparison(sources["pdc_comparison"]),
        "pdc_per_class": build_pdc_per_class(sources["pdc_per_class"]),
        "attribution_stability": build_attribution_stability(
            sources["attribution_stability"]
        ),
        "marker_enrichment": build_marker_enrichment(
            sources["marker_enrichment"]
        ),
    }
    manifest = build_source_manifest(root)
    outputs: Dict[str, Path] = {}
    for name, frame in tables.items():
        outputs[name] = _write_mirrored_frame(
            root,
            OUTPUT_FILENAMES[name],
            frame,
        )
    outputs["manifest_csv"] = _write_mirrored_frame(
        root,
        OUTPUT_FILENAMES["manifest_csv"],
        manifest,
    )
    manifest_payload = {
        "version": "V33",
        "date": DATE,
        "complete": True,
        "legacy_performance_fallback": False,
        "source_artifacts": manifest.to_dict(orient="records"),
    }
    outputs["manifest_json"] = _write_mirrored_bytes(
        root,
        OUTPUT_FILENAMES["manifest_json"],
        (json.dumps(manifest_payload, indent=2) + "\n").encode("utf-8"),
    )
    outputs["latex_bundle"] = _write_mirrored_bytes(
        root,
        OUTPUT_FILENAMES["latex_bundle"],
        render_latex_bundle(tables).encode("utf-8"),
    )
    for role, fragment in render_latex_fragments(tables).items():
        outputs[role] = _write_mirrored_bytes(
            root,
            OUTPUT_FILENAMES[role],
            fragment.encode("utf-8"),
        )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    outputs = generate_manuscript_tables(args.root)
    for role, path in outputs.items():
        print(f"{role}: {path}")


if __name__ == "__main__":
    main()
