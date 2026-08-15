#!/usr/bin/env python
"""Render the three artifact-backed cell-level audit records as Figure 3."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
DATE = "2026-08-14"
SOURCE = ROOT / f"results/tables/mosaic_n_v8_audit_case_study_records_{DATE}.csv"
ROLES = [
    ("agreement_correct", "Agreement record", "#2a9d8f"),
    ("modality_conflict", "Modality conflict", "#e76f51"),
    ("hsr_high_delta_or_changed", "Bounded HSR action", "#e9c46a"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean(value: object) -> str:
    if pd.isna(value):
        return "not recorded"
    return str(value)


def _branch_text(row: pd.Series) -> str:
    values = []
    for name, column in (("RNA", "rna_prediction"), ("ADT", "adt_prediction"), ("fusion", "fusion_prediction")):
        value = _clean(row[column])
        values.append(f"{name}: {value}")
    return "\n".join(values)


def _draw_card(ax, row: pd.Series, title: str, color: str, panel: str) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    card = FancyBboxPatch(
        (0.01, 0.02),
        0.98,
        0.96,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.2,
        edgecolor="#b7c1cc",
        facecolor="#fbfcfd",
    )
    ax.add_patch(card)
    ax.add_patch(plt.Rectangle((0.01, 0.94), 0.98, 0.04, color=color, transform=ax.transAxes, clip_on=False))
    ax.text(0.04, 0.895, f"{panel}  {title}", fontsize=10.2, fontweight="bold", color="#1f2933")
    ax.text(
        0.04,
        0.825,
        f"Donor {row['test_donor']}  |  seed {int(row['seed'])}\n{textwrap.fill(str(row['cell_id']), width=22)}",
        fontsize=7.2,
        color="#52606d",
        va="top",
    )
    ax.text(0.04, 0.675, "True label", fontsize=7.2, color="#697586")
    ax.text(0.96, 0.675, _clean(row["label"]), fontsize=8.0, color="#1f2933", ha="right", fontweight="bold")
    ax.text(0.04, 0.605, "Final output", fontsize=7.2, color="#697586")
    final_text = _clean(row["prediction"])
    final_color = "#2a9d8f" if not bool(row["error"]) else "#c0392b"
    ax.text(0.96, 0.605, final_text, fontsize=8.0, color=final_color, ha="right", fontweight="bold")
    ax.text(0.04, 0.515, "Branch outputs", fontsize=7.2, color="#697586")
    ax.text(0.04, 0.475, _branch_text(row), fontsize=6.7, color="#1f2933", va="top")

    case_type = row.get("case_type", row.name)
    if case_type == "agreement_correct":
        metrics = [
            f"uncertainty = {float(row['final_uncertainty']):.5f}",
            "branch disagreement = 0",
            "HSR action = none recorded",
        ]
        outcome = "correct agreement; review record, not a guarantee"
    elif case_type == "modality_conflict":
        metrics = [
            f"uncertainty = {float(row['final_uncertainty']):.3f}",
            f"branch disagreement = {float(row['branch_disagreement']):.3f}",
            "HSR group = not applicable",
        ]
        outcome = "conflict flag; final output is wrong"
    else:
        metrics = [
            f"base -> final = {_clean(row['base_prediction'])} -> {_clean(row['prediction'])}",
            f"gate = {float(row['hsr_gate']):.4f}",
            f"||delta||2 = {float(row['hsr_delta_norm']):.3f}",
        ]
        outcome = "sparse local action; final output is wrong"
    ax.text(0.04, 0.285, "\n".join(metrics), fontsize=7.0, color="#1f2933", va="top")
    ax.text(
        0.04,
        0.075,
        outcome,
        fontsize=7.0,
        color=final_color if bool(row["error"]) else "#256d5c",
        va="bottom",
        fontweight="bold",
    )


def _draw_compact_card(ax, row: pd.Series, title: str, color: str, panel: str) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    card = FancyBboxPatch(
        (0.01, 0.02),
        0.98,
        0.96,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=0.8,
        edgecolor="#b7c1cc",
        facecolor="#fbfcfd",
    )
    ax.add_patch(card)
    ax.add_patch(plt.Rectangle((0.01, 0.94), 0.98, 0.04, color=color, transform=ax.transAxes, clip_on=False))
    ax.text(0.04, 0.875, f"{panel}  {title}", fontsize=8.3, fontweight="bold", color="#1f2933")
    ax.text(
        0.04,
        0.785,
        f"Donor {row['test_donor']} | seed {int(row['seed'])}\n{textwrap.fill(str(row['cell_id']), width=34)}",
        fontsize=5.4,
        color="#52606d",
        va="top",
    )
    ax.text(0.04, 0.605, "True", fontsize=5.8, color="#697586")
    ax.text(0.96, 0.605, _clean(row["label"]), fontsize=6.6, color="#1f2933", ha="right", fontweight="bold")
    final_color = "#2a9d8f" if not bool(row["error"]) else "#c0392b"
    ax.text(0.04, 0.535, "Final", fontsize=5.8, color="#697586")
    ax.text(0.96, 0.535, _clean(row["prediction"]), fontsize=6.6, color=final_color, ha="right", fontweight="bold")
    case_type = row.get("case_type", row.name)
    branch = (
        f"RNA={_clean(row['rna_prediction'])}; ADT={_clean(row['adt_prediction'])}; "
        f"fusion={_clean(row['fusion_prediction'])}"
    )
    ax.text(0.04, 0.445, "Branches", fontsize=5.8, color="#697586")
    ax.text(0.04, 0.405, textwrap.fill(branch, width=42), fontsize=5.3, color="#1f2933", va="top")
    if case_type == "agreement_correct":
        metrics = f"uncertainty={float(row['final_uncertainty']):.5f} | conflict=0\nHSR action: none recorded"
        outcome = "correct agreement; review record, not a guarantee"
    elif case_type == "modality_conflict":
        metrics = (
            f"uncertainty={float(row['final_uncertainty']):.3f} | "
            f"conflict={float(row['branch_disagreement']):.3f}\nHSR group: not applicable"
        )
        outcome = "conflict flag; final output is wrong"
    else:
        metrics = (
            f"base -> final: {_clean(row['base_prediction'])} -> {_clean(row['prediction'])}\n"
            f"gate={float(row['hsr_gate']):.4f} | ||delta||2={float(row['hsr_delta_norm']):.3f}"
        )
        outcome = "sparse local action; final output is wrong"
    ax.text(0.04, 0.270, metrics, fontsize=5.5, color="#1f2933", va="top")
    ax.text(0.04, 0.050, outcome, fontsize=5.5, color=final_color if bool(row["error"]) else "#256d5c", va="bottom", fontweight="bold")


def build_figure() -> tuple[Path, Path, dict]:
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    frame = pd.read_csv(SOURCE)
    expected = [role for role, _, _ in ROLES]
    if frame["case_type"].tolist() != sorted(expected):
        frame = frame.set_index("case_type").loc[sorted(expected)].reset_index()
    if frame["case_type"].tolist() != sorted(expected):
        raise ValueError("audit case source does not contain the three expected roles")

    figure_dir = ROOT / "manufacture/mosaic_n_bioinformatics_manuscript_v1/oup-authoring-template/Fig"
    output_dir = ROOT / "output/figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = figure_dir / "Fig3_MOSAIC_audit_cases.pdf"
    png_path = figure_dir / "Fig3_MOSAIC_audit_cases.png"
    narrow_pdf_path = figure_dir / "Fig3_MOSAIC_audit_cases_singlecol.pdf"
    narrow_png_path = figure_dir / "Fig3_MOSAIC_audit_cases_singlecol.png"

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.55), dpi=220)
    fig.patch.set_facecolor("white")
    for axis, (role, title, color), (_, row) in zip(axes, ROLES, frame.set_index("case_type").loc[expected].iterrows()):
        _draw_compact_card(axis, row, title, color, f"({chr(97 + expected.index(role))})")
    fig.suptitle(
        "Cell-level audit records expose agreement, conflict and bounded local action",
        fontsize=11.2,
        fontweight="bold",
        color="#1f2933",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965), w_pad=1.0)
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    fig.savefig(png_path, bbox_inches="tight", facecolor="white", dpi=240)
    fig.savefig(output_dir / f"mosaic_n_v8_audit_cases_{DATE}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(output_dir / f"mosaic_n_v8_audit_cases_{DATE}.png", bbox_inches="tight", facecolor="white", dpi=240)
    plt.close(fig)

    narrow_fig, narrow_axes = plt.subplots(3, 1, figsize=(3.35, 4.35), dpi=240)
    narrow_fig.patch.set_facecolor("white")
    for axis, (role, title, color), (_, row) in zip(
        narrow_axes,
        ROLES,
        frame.set_index("case_type").loc[expected].iterrows(),
    ):
        _draw_compact_card(axis, row, title, color, f"({chr(97 + expected.index(role))})")
    narrow_fig.suptitle(
        "Cell-level audit records",
        fontsize=10.0,
        fontweight="bold",
        color="#1f2933",
        y=0.998,
    )
    narrow_fig.tight_layout(rect=(0, 0, 1, 0.982), h_pad=0.22)
    narrow_fig.savefig(narrow_pdf_path, bbox_inches="tight", facecolor="white")
    narrow_fig.savefig(narrow_png_path, bbox_inches="tight", facecolor="white", dpi=260)
    narrow_fig.savefig(
        output_dir / f"mosaic_n_v8_audit_cases_singlecol_{DATE}.pdf",
        bbox_inches="tight",
        facecolor="white",
    )
    narrow_fig.savefig(
        output_dir / f"mosaic_n_v8_audit_cases_singlecol_{DATE}.png",
        bbox_inches="tight",
        facecolor="white",
        dpi=260,
    )
    plt.close(narrow_fig)

    metadata = {
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": sha256(SOURCE),
        "roles": expected,
        "figure_pdf": str(pdf_path.relative_to(ROOT)),
        "figure_png": str(png_path.relative_to(ROOT)),
        "figure_singlecol_pdf": str(narrow_pdf_path.relative_to(ROOT)),
        "figure_singlecol_png": str(narrow_png_path.relative_to(ROOT)),
        "interpretation": "three artifact-backed records; structured review evidence, not automatic correctness",
    }
    metadata_path = output_dir / f"mosaic_n_v8_audit_cases_{DATE}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return pdf_path, png_path, metadata


if __name__ == "__main__":
    pdf, png, metadata = build_figure()
    print(json.dumps({"pdf": str(pdf), "png": str(png), "source_sha256": metadata["source_sha256"]}, indent=2))
