#!/usr/bin/env python
"""Plot compact case-study evidence for unknown failure and CD8 boundary gaps."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "manufacture/mosaic_n_bioinformatics_manuscript_v1/oup-authoring-template/Fig"
OUT_PDF = FIG_DIR / "FigS_case_study_marker_limits.pdf"
OUT_PNG = FIG_DIR / "FigS_case_study_marker_limits.png"


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), constrained_layout=True)

    ax = axes[0]
    metrics = ["Unknown\nrecall", "AUROC", "Parent-safe\nrate"]
    b_values = [0.1285, 0.4258, 0.0723]
    reference = [0.4032, 0.5948, 0.2259]
    x = np.arange(len(metrics))
    width = 0.36
    ax.bar(x - width / 2, b_values, width, color="#8C1D40", label="B naive lambda")
    ax.bar(x + width / 2, reference, width, color="#2F6F9F", label="Mean target")
    ax.axhline(0.5, color="#666666", lw=0.8, ls="--")
    ax.set_ylim(0, 0.8)
    ax.set_xticks(x, metrics)
    ax.set_ylabel("Score")
    ax.set_title("A. Light-chain unknown failure")
    ax.text(0.02, 0.74, "ADT kappa/lambda marker: 0/8 folds", transform=ax.transAxes, ha="left", va="top")
    ax.legend(frameon=False, loc="upper right", fontsize=7)

    ax = axes[1]
    classes = ["cd8_cm", "cd8_em", "cd8_emra", "cd8_n", "monocyte"]
    mosaic = np.array([0.8246, 0.8825, 0.8863, 0.9260, 0.9970])
    mmochi = np.array([0.8851, 0.9122, 0.9493, 0.9734, 0.9970])
    gaps = mmochi - mosaic
    colors = ["#D95F02" if cls.startswith("cd8") else "#4D4D4D" for cls in classes]
    ax.bar(np.arange(len(classes)), gaps, color=colors)
    ax.axhline(0, color="#333333", lw=0.8)
    ax.set_ylim(0, 0.075)
    ax.set_xticks(np.arange(len(classes)), classes, rotation=25, ha="right")
    ax.set_ylabel("MMoCHi F1 - MOSAIC F1")
    ax.set_title("B. CD8 boundary gap on PDC101")
    for idx, value in enumerate(gaps):
        ax.text(idx, value + 0.002, f"{value:.3f}", ha="center", va="bottom", fontsize=7)

    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    print(OUT_PDF)
    print(OUT_PNG)


if __name__ == "__main__":
    main()
