#!/usr/bin/env python
"""Plot the V33 evidence figure from the validated V33 manuscript tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_v33_manuscript_tables import (
    DATE,
    OUTPUT_FILENAMES,
    ROOT,
    SOURCE_FILES,
)


FIGURE_BASENAME = "mosaic_n_v33_evidence_figure"
FIGURE_SOURCE_CSV = f"{FIGURE_BASENAME}_sources_{DATE}.csv"
FIGURE_SOURCE_JSON = f"{FIGURE_BASENAME}_sources_{DATE}.json"

METRIC_LABELS = {
    "accuracy": "ACC",
    "weighted_f1": "W-F1",
    "macro_f1": "M-F1",
    "balanced_accuracy": "Balanced ACC",
}
SCENARIO_ORDER = [
    "full",
    "random_10",
    "random_30",
    "random_50",
    "random_70",
    "marker_memory",
    "marker_tcell",
    "rna_only",
]
SCENARIO_LABELS = {
    "full": "Full",
    "random_10": "10%",
    "random_30": "30%",
    "random_50": "50%",
    "random_70": "70%",
    "marker_memory": "Memory\nmarkers",
    "marker_tcell": "T-cell\nmarkers",
    "rna_only": "RNA\nonly",
}
SCORE_LABELS = {
    "one_minus_max_probability": "Max prob.",
    "one_minus_margin": "Margin",
    "energy": "Energy",
}
SCORE_ORDER = [
    "one_minus_max_probability",
    "one_minus_margin",
    "energy",
]
TARGET_ORDER = [
    "B naive lambda",
    "NK_3",
    "gdT_2",
    "CD4 TCM_1",
    "CD8 TEM_4",
]
PDC_CLASS_ORDER = [
    "cd4_cm",
    "cd4_em",
    "cd4_n",
    "cd8_cm",
    "cd8_em",
    "cd8_emra",
    "cd8_n",
    "monocyte",
]
COLORS = {
    "ink": "#2B2B2B",
    "teal": "#238B8E",
    "teal_dark": "#146C70",
    "vermilion": "#C65D4B",
    "ochre": "#C79528",
    "navy": "#365C8D",
    "gray": "#777777",
    "light_gray": "#D8D8D8",
    "grid": "#E5E7E9",
}


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.0,
            "axes.titlesize": 8.2,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.4,
            "legend.fontsize": 6.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _required_table_paths(root: Path) -> Dict[str, Path]:
    keys = [
        "paired_statistics",
        "panel_robustness",
        "pdc_comparison",
        "pdc_per_class",
        "attribution_stability",
        "marker_enrichment",
    ]
    paths = {
        key: root / "results/tables" / OUTPUT_FILENAMES[key]
        for key in keys
    }
    paths["manifest_json"] = (
        root / "results/tables" / OUTPUT_FILENAMES["manifest_json"]
    )
    paths["unknown_metrics"] = root / SOURCE_FILES["unknown_metrics"]
    return paths


def load_table_package(root: Path = ROOT) -> Dict[str, object]:
    root = Path(root)
    paths = _required_table_paths(root)
    missing = [
        str(path.relative_to(root))
        for path in paths.values()
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "validated V33 manuscript tables are missing; run "
            f"build_v33_manuscript_tables.py first: {missing}"
        )
    manifest = json.loads(paths["manifest_json"].read_text(encoding="utf-8"))
    if (
        manifest.get("version") != "V33"
        or manifest.get("complete") is not True
        or manifest.get("legacy_performance_fallback") is not False
    ):
        raise ValueError("V33 table manifest is incomplete or permits legacy fallback")
    tables: Dict[str, object] = {"manifest": manifest}
    for key, path in paths.items():
        if key != "manifest_json":
            tables[key] = pd.read_csv(path)
    return tables


def _assert_provenance(frame: pd.DataFrame, name: str) -> None:
    required = {"protocol", "caveat", "source_artifact"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} missing provenance columns: {sorted(missing)}")
    for column in sorted(required):
        if frame[column].fillna("").astype(str).str.strip().eq("").any():
            raise ValueError(f"{name} contains empty {column}")
    if not frame["source_artifact"].astype(str).str.contains(
        "v33",
        case=False,
        regex=False,
    ).all():
        raise ValueError(f"{name} contains a non-V33 source artifact")


def build_figure_data(tables: Dict[str, object]) -> pd.DataFrame:
    paired = tables["paired_statistics"]
    panel = tables["panel_robustness"]
    unknown = tables["unknown_metrics"]
    pdc = tables["pdc_comparison"]
    pdc_per_class = tables["pdc_per_class"]
    attribution = tables["attribution_stability"]
    enrichment = tables["marker_enrichment"]
    for name, frame in [
        ("paired_statistics", paired),
        ("panel_robustness", panel),
        ("pdc_comparison", pdc),
        ("pdc_per_class", pdc_per_class),
        ("attribution_stability", attribution),
        ("marker_enrichment", enrichment),
    ]:
        _assert_provenance(frame, name)

    panel_a = paired[
        paired["reference"].eq("mosaic_full")
        & paired["comparator"].eq("mlp")
        & paired["metric"].isin(METRIC_LABELS)
    ].copy()
    panel_a["panel"] = "a"

    panel_b = paired[
        (
            paired["reference"].eq("mosaic_full")
            & paired["comparator"].isin(["mosaic_no_hsr", "mosaic_no_kd"])
        )
        | (
            paired["reference"].eq("inference::margin_gate_hsr")
            & paired["comparator"].eq("inference::uniform_fusion")
        )
    ].copy()
    panel_b = panel_b[panel_b["metric"].isin(
        ["accuracy", "weighted_f1", "macro_f1"]
    )]
    panel_b["panel"] = "b"

    panel_c = panel[
        panel["metric"].isin(["weighted_f1", "macro_f1", "ece"])
        & panel["scenario"].isin(SCENARIO_ORDER)
    ].copy()
    panel_c["panel"] = "c"

    required_unknown = {
        "target_label",
        "score",
        "known_coverage_target",
        "unknown_recall",
        "n_test_unknown",
    }
    missing_unknown = required_unknown.difference(unknown.columns)
    if missing_unknown:
        raise ValueError(
            "unknown_metrics missing figure columns: "
            + ", ".join(sorted(missing_unknown))
        )
    panel_d = (
        unknown[
            unknown["known_coverage_target"].astype(float).round(2).eq(0.80)
            & unknown["target_label"].isin(TARGET_ORDER)
            & unknown["score"].isin(SCORE_ORDER)
        ]
        .groupby(["target_label", "score"], as_index=False)
        .agg(
            unknown_recall=("unknown_recall", "mean"),
            n_unknown=("n_test_unknown", "max"),
        )
    )
    panel_d["statistical_unit"] = "leave-class-out target after seed averaging"
    panel_d["protocol"] = "validation-selected reject at target known coverage 0.80"
    panel_d["caveat"] = (
        "Target support and difficulty vary; these are pseudo-unknown stress tests."
    )
    panel_d["source_artifact"] = SOURCE_FILES["unknown_metrics"]
    panel_d["panel"] = "d"

    panel_e_aggregate = pdc.copy()
    panel_e_aggregate["record_type"] = "aggregate"
    panel_e_per_class = pdc_per_class.copy()
    panel_e_per_class["record_type"] = "per_class"
    panel_e = pd.concat(
        [panel_e_aggregate, panel_e_per_class],
        ignore_index=True,
        sort=False,
    )
    panel_e["panel"] = "e"

    panel_f = attribution.merge(
        enrichment[
            [
                "class_label",
                "modality",
                "observed_fraction",
                "permutation_pvalue",
                "source_artifact",
            ]
        ],
        on=["class_label", "modality"],
        how="inner",
        validate="one_to_one",
        suffixes=("", "_enrichment"),
    )
    panel_f["source_artifact"] = (
        panel_f["source_artifact"].astype(str)
        + ";"
        + panel_f["source_artifact_enrichment"].astype(str)
    )
    panel_f = panel_f.drop(columns=["source_artifact_enrichment"])
    panel_f["panel"] = "f"

    return pd.concat(
        [panel_a, panel_b, panel_c, panel_d, panel_e, panel_f],
        ignore_index=True,
        sort=False,
    )


def _panel_label(
    axis: plt.Axes,
    label: str,
    title: str,
    *,
    grid_axis: Optional[str] = None,
) -> None:
    axis.set_title(title, loc="left", pad=7, fontweight="bold")
    axis.text(
        -0.13,
        1.07,
        label,
        transform=axis.transAxes,
        fontsize=9.5,
        fontweight="bold",
        va="top",
    )
    if grid_axis:
        axis.grid(
            axis=grid_axis,
            color=COLORS["grid"],
            linewidth=0.6,
            zorder=0,
        )


def _format_pvalue(value: float) -> str:
    if value < 0.001:
        return r"$P<0.001$"
    return rf"$P={value:.3f}$"


def _annotated_heatmap(
    axis: plt.Axes,
    matrix: np.ndarray,
    row_labels: Iterable[str],
    column_labels: Iterable[str],
    *,
    cmap: str,
    vmin: float,
    vmax: float,
    value_format: str,
    text_threshold: Optional[float] = None,
    light_text_above: bool = True,
    suffixes: Optional[np.ndarray] = None,
) -> None:
    axis.imshow(
        matrix,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
        interpolation="nearest",
    )
    axis.set_xticks(np.arange(matrix.shape[1]))
    axis.set_xticklabels(list(column_labels))
    axis.set_yticks(np.arange(matrix.shape[0]))
    axis.set_yticklabels(list(row_labels))
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            suffix = "" if suffixes is None else str(suffixes[row, column])
            threshold = (
                (vmin + vmax) / 2
                if text_threshold is None
                else text_threshold
            )
            use_light_text = (
                value >= threshold
                if light_text_above
                else value < threshold
            )
            color = "white" if use_light_text else COLORS["ink"]
            axis.text(
                column,
                row,
                format(value, value_format) + suffix,
                ha="center",
                va="center",
                fontsize=5.9,
                color=color,
            )
    axis.set_xticks(
        np.arange(-0.5, matrix.shape[1], 1),
        minor=True,
    )
    axis.set_yticks(
        np.arange(-0.5, matrix.shape[0], 1),
        minor=True,
    )
    axis.grid(which="minor", color="white", linewidth=1.0)
    axis.tick_params(which="minor", bottom=False, left=False)
    for spine in axis.spines.values():
        spine.set_visible(False)


def _plot_main_comparison(axis: plt.Axes, frame: pd.DataFrame) -> None:
    metric_order = [
        "accuracy",
        "weighted_f1",
        "macro_f1",
        "balanced_accuracy",
    ]
    subset = frame.set_index("metric").loc[metric_order]
    y = np.arange(len(metric_order))[::-1]
    mean = subset["mean_difference"].to_numpy(dtype=float) * 100
    low = subset["difference_ci95_low"].to_numpy(dtype=float) * 100
    high = subset["difference_ci95_high"].to_numpy(dtype=float) * 100
    axis.axvline(0, color=COLORS["gray"], linewidth=0.8, zorder=1)
    axis.errorbar(
        mean,
        y,
        xerr=np.vstack([mean - low, high - mean]),
        fmt="o",
        color=COLORS["teal_dark"],
        ecolor=COLORS["teal"],
        markersize=4.2,
        elinewidth=1.3,
        capsize=2.5,
        zorder=3,
    )
    axis.set_yticks(y)
    axis.set_yticklabels([METRIC_LABELS[metric] for metric in metric_order])
    axis.set_xlabel("Full − MLP (percentage points)")
    left = min(-0.35, float(low.min()) - 0.25)
    right = float(high.max()) + 1.45
    axis.set_xlim(left, right)
    for position, row in zip(y, subset.itertuples()):
        annotation = (
            f"{int(row.positive_donor_count)}/8; "
            f"{_format_pvalue(float(row.paired_t_pvalue))}"
        )
        axis.text(
            float(row.difference_ci95_high) * 100 + 0.12,
            position,
            annotation,
            va="center",
            fontsize=5.7,
            color=COLORS["ink"],
        )
    axis.tick_params(axis="y", length=0)
    _panel_label(
        axis,
        "a",
        "Primary donor-paired effect",
        grid_axis="x",
    )


def _plot_module_ablation(axis: plt.Axes, frame: pd.DataFrame) -> None:
    comparisons = [
        ("mosaic_full", "mosaic_no_hsr", "Full − No HSR"),
        ("mosaic_full", "mosaic_no_kd", "Full − No KD"),
        (
            "inference::margin_gate_hsr",
            "inference::uniform_fusion",
            "Gate+HSR − Uniform",
        ),
    ]
    metric_order = ["accuracy", "weighted_f1", "macro_f1"]
    rows = []
    for reference, comparator, label in comparisons:
        block = (
            frame[
                frame["reference"].eq(reference)
                & frame["comparator"].eq(comparator)
            ]
            .set_index("metric")
            .loc[metric_order]
        )
        for metric, row in block.iterrows():
            rows.append(
                {
                    "label": f"{label} · {METRIC_LABELS[metric]}",
                    "mean": float(row["mean_difference"]) * 100,
                    "low": float(row["difference_ci95_low"]) * 100,
                    "high": float(row["difference_ci95_high"]) * 100,
                }
            )
    y = np.arange(len(rows))[::-1]
    axis.axvline(0, color=COLORS["gray"], linewidth=0.8, zorder=1)
    for position, row in zip(y, rows):
        color = (
            COLORS["vermilion"]
            if row["mean"] < 0
            else COLORS["teal_dark"]
        )
        axis.errorbar(
            row["mean"],
            position,
            xerr=np.array(
                [[row["mean"] - row["low"]], [row["high"] - row["mean"]]]
            ),
            fmt="o",
            color=color,
            ecolor=color,
            markersize=3.4,
            elinewidth=1.0,
            capsize=2.0,
            zorder=3,
        )
        axis.text(
            row["high"] + 0.04,
            position,
            f"{row['mean']:+.2f}",
            va="center",
            fontsize=5.4,
            color=COLORS["ink"],
        )
    axis.axhline(5.5, color=COLORS["light_gray"], linewidth=0.7)
    axis.axhline(2.5, color=COLORS["light_gray"], linewidth=0.7)
    axis.set_yticks(y)
    axis.set_yticklabels([row["label"] for row in rows], fontsize=5.5)
    axis.set_xlabel("Reference − comparator (percentage points)")
    low = min(row["low"] for row in rows)
    high = max(row["high"] for row in rows)
    axis.set_xlim(min(-1.05, low - 0.1), max(0.35, high + 0.3))
    axis.tick_params(axis="y", length=0)
    _panel_label(
        axis,
        "b",
        "Direct architecture ablations",
        grid_axis="x",
    )


def _plot_panel_robustness(axis: plt.Axes, frame: pd.DataFrame) -> None:
    metric_order = ["weighted_f1", "macro_f1", "ece"]
    pivot = (
        frame.pivot(index="scenario", columns="metric", values="mean")
        .loc[SCENARIO_ORDER, metric_order]
        .astype(float)
    )
    effects = pivot.copy()
    effects["weighted_f1"] = (
        pivot["weighted_f1"] - pivot.loc["full", "weighted_f1"]
    )
    effects["macro_f1"] = (
        pivot["macro_f1"] - pivot.loc["full", "macro_f1"]
    )
    effects["ece"] = pivot.loc["full", "ece"] - pivot["ece"]
    matrix = effects.to_numpy(dtype=float) * 100
    scale = max(1.0, float(np.abs(matrix).max()))
    _annotated_heatmap(
        axis,
        matrix,
        [SCENARIO_LABELS[value].replace("\n", " ") for value in SCENARIO_ORDER],
        [r"$\Delta$W-F1", r"$\Delta$M-F1", r"$-\Delta$ECE"],
        cmap="RdBu",
        vmin=-scale,
        vmax=scale,
        value_format="+.1f",
        text_threshold=scale + 1,
    )
    axis.set_xlabel("Percentage-point change from full; higher is better")
    _panel_label(axis, "c", "Incomplete-protein stress test")


def _plot_unknown_reject(axis: plt.Axes, frame: pd.DataFrame) -> None:
    pivot = (
        frame.pivot(
            index="target_label",
            columns="score",
            values="unknown_recall",
        )
        .loc[TARGET_ORDER, SCORE_ORDER]
        .astype(float)
    )
    support = (
        frame.groupby("target_label")["n_unknown"]
        .max()
        .loc[TARGET_ORDER]
        .astype(int)
    )
    labels = [
        f"{target} (n={support.loc[target]})"
        for target in TARGET_ORDER
    ]
    matrix = pivot.to_numpy(dtype=float)
    _annotated_heatmap(
        axis,
        matrix,
        labels,
        [SCORE_LABELS[value] for value in SCORE_ORDER],
        cmap="YlGnBu",
        vmin=0.0,
        vmax=max(0.65, float(matrix.max())),
        value_format=".2f",
        text_threshold=0.43,
    )
    axis.set_xlabel("Unknown recall at target known coverage 0.80")
    _panel_label(axis, "d", "Leave-class-out target heterogeneity")


def _plot_pdc_comparison(axis: plt.Axes, frame: pd.DataFrame) -> None:
    aggregate = frame[frame["record_type"].eq("aggregate")]
    aggregate_pivot = aggregate.pivot(
        index="metric",
        columns="method",
        values="mean",
    )
    rows = [
        (
            METRIC_LABELS[metric],
            float(aggregate_pivot.loc[metric, "MMoCHi"])
            - float(aggregate_pivot.loc[metric, "MOSAIC-N"]),
            "aggregate",
        )
        for metric in ["accuracy", "weighted_f1", "macro_f1"]
    ]
    per_class = frame[frame["record_type"].eq("per_class")]
    class_pivot = per_class.pivot(
        index="class_label",
        columns="method",
        values="mean_f1",
    )
    supports = (
        per_class.groupby("class_label")["support"]
        .max()
        .astype(int)
    )
    rows.extend(
        (
            f"{class_label} (n={supports.loc[class_label]})",
            float(class_pivot.loc[class_label, "MMoCHi"])
            - float(class_pivot.loc[class_label, "MOSAIC-N"]),
            "per_class",
        )
        for class_label in PDC_CLASS_ORDER
    )
    y = np.arange(len(rows))[::-1]
    values = np.array([row[1] for row in rows], dtype=float) * 100
    axis.axvline(0, color=COLORS["gray"], linewidth=0.8, zorder=1)
    for position, value, row in zip(y, values, rows):
        color = COLORS["vermilion"] if value > 0 else COLORS["teal"]
        marker = "s" if row[2] == "aggregate" else "o"
        axis.scatter(
            value,
            position,
            s=20 if marker == "s" else 14,
            marker=marker,
            color=color,
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )
        axis.hlines(
            position,
            0,
            value,
            color=color,
            linewidth=0.9,
            zorder=2,
        )
        axis.text(
            value + (0.3 if value >= 0 else -0.3),
            position,
            f"{value:+.1f}",
            ha="left" if value >= 0 else "right",
            va="center",
            fontsize=5.2,
        )
    axis.axhline(7.5, color=COLORS["light_gray"], linewidth=0.8)
    axis.set_yticks(y)
    axis.set_yticklabels([row[0] for row in rows], fontsize=5.5)
    axis.set_xlabel(
        "MMoCHi − MOSAIC-N (percentage points; point estimates)"
    )
    minimum = min(-1.0, float(values.min()) - 1.0)
    maximum = max(1.5, float(values.max()) + 1.3)
    axis.set_xlim(minimum, maximum)
    axis.tick_params(axis="y", length=0)
    _panel_label(
        axis,
        "e",
        "PDC101 external method boundary",
        grid_axis="x",
    )


def _plot_attribution(axis: plt.Axes, frame: pd.DataFrame) -> None:
    class_order = [
        "CD4 TCM_3",
        "CD4 TEM_3",
        "CD4 TEM_4",
        "CD8 Naive",
        "CD8 Naive_2",
        "CD8 TCM_1",
    ]
    row_index = [
        (class_label, modality)
        for class_label in class_order
        for modality in ["RNA", "ADT"]
    ]
    indexed = frame.set_index(["class_label", "modality"]).loc[row_index]
    columns = [
        "mean_spearman",
        "mean_top20_jaccard",
        "observed_fraction",
    ]
    matrix = indexed[columns].to_numpy(dtype=float)
    suffixes = np.full(matrix.shape, "", dtype=object)
    suffixes[:, 2] = np.where(
        indexed["permutation_pvalue"].to_numpy(dtype=float) < 0.05,
        "*",
        "",
    )
    labels = [f"{cell_type} · {modality}" for cell_type, modality in row_index]
    _annotated_heatmap(
        axis,
        matrix,
        labels,
        [r"Rank $\rho$", "Top-20 J", "Marker overlap"],
        cmap="cividis",
        vmin=0.0,
        vmax=1.0,
        value_format=".2f",
        text_threshold=0.45,
        light_text_above=False,
        suffixes=suffixes,
    )
    for tick, (_, modality) in zip(axis.get_yticklabels(), row_index):
        tick.set_color(
            COLORS["navy"] if modality == "RNA" else COLORS["ochre"]
        )
    axis.set_xlabel("* unadjusted marker-enrichment permutation P<0.05")
    _panel_label(axis, "f", "Cross-donor feature attribution")


def build_figure(root: Path = ROOT) -> Tuple[plt.Figure, pd.DataFrame]:
    configure_matplotlib()
    tables = load_table_package(root)
    figure_data = build_figure_data(tables)

    figure = plt.figure(figsize=(8.0, 8.4))
    grid = figure.add_gridspec(
        3,
        2,
        height_ratios=[0.92, 1.02, 1.35],
        hspace=0.48,
        wspace=0.52,
    )
    axes = [
        figure.add_subplot(grid[0, 0]),
        figure.add_subplot(grid[0, 1]),
        figure.add_subplot(grid[1, 0]),
        figure.add_subplot(grid[1, 1]),
        figure.add_subplot(grid[2, 0]),
        figure.add_subplot(grid[2, 1]),
    ]
    _plot_main_comparison(axes[0], figure_data[figure_data["panel"].eq("a")])
    _plot_module_ablation(axes[1], figure_data[figure_data["panel"].eq("b")])
    _plot_panel_robustness(axes[2], figure_data[figure_data["panel"].eq("c")])
    _plot_unknown_reject(axes[3], figure_data[figure_data["panel"].eq("d")])
    _plot_pdc_comparison(axes[4], figure_data[figure_data["panel"].eq("e")])
    _plot_attribution(axes[5], figure_data[figure_data["panel"].eq("f")])
    figure.subplots_adjust(
        left=0.13,
        right=0.985,
        top=0.965,
        bottom=0.055,
        hspace=0.50,
        wspace=0.55,
    )
    return figure, figure_data


def _write_mirrored_bytes(
    root: Path,
    result_relative: Path,
    mirror_relative: Path,
    content: bytes,
) -> Path:
    result_path = root / result_relative
    mirror_path = root / mirror_relative
    for path in [result_path, mirror_path]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return result_path


def generate_evidence_figure(
    root: Path = ROOT,
    *,
    dpi: int = 600,
) -> Dict[str, Path]:
    root = Path(root)
    figure, figure_data = build_figure(root)
    outputs: Dict[str, Path] = {}
    result_dir = root / "results/figures"
    mirror_dir = root / "output/figures"
    result_dir.mkdir(parents=True, exist_ok=True)
    mirror_dir.mkdir(parents=True, exist_ok=True)
    try:
        for suffix in [".pdf", ".svg", ".png"]:
            result_path = result_dir / f"{FIGURE_BASENAME}_{DATE}{suffix}"
            save_kwargs = {"bbox_inches": "tight"}
            if suffix == ".png":
                save_kwargs["dpi"] = dpi
            figure.savefig(result_path, **save_kwargs)
            mirror_path = mirror_dir / result_path.name
            mirror_path.write_bytes(result_path.read_bytes())
            outputs[suffix.lstrip(".")] = result_path
    finally:
        plt.close(figure)

    source_csv = figure_data.to_csv(index=False).encode("utf-8")
    outputs["source_manifest_csv"] = _write_mirrored_bytes(
        root,
        Path("results/tables") / FIGURE_SOURCE_CSV,
        Path("output/tables") / FIGURE_SOURCE_CSV,
        source_csv,
    )
    source_payload = {
        "version": "V33",
        "date": DATE,
        "figure": f"{FIGURE_BASENAME}_{DATE}",
        "panels": {
            panel: sorted(
                set(
                    figure_data.loc[
                        figure_data["panel"].eq(panel),
                        "source_artifact",
                    ].astype(str)
                )
            )
            for panel in ["a", "b", "c", "d", "e", "f"]
        },
        "protocol_and_caveat_embedded_in_csv": True,
        "legacy_performance_fallback": False,
    }
    outputs["source_manifest_json"] = _write_mirrored_bytes(
        root,
        Path("results/tables") / FIGURE_SOURCE_JSON,
        Path("output/tables") / FIGURE_SOURCE_JSON,
        (json.dumps(source_payload, indent=2) + "\n").encode("utf-8"),
    )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    outputs = generate_evidence_figure(args.root, dpi=args.dpi)
    for role, path in outputs.items():
        print(f"{role}: {path}")


if __name__ == "__main__":
    main()
