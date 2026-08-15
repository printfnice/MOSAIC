#!/usr/bin/env python
"""Build manuscript supplement table for full-training PBMC donor baselines."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATE = "2026-07-29"
SOURCES = [
    ROOT / "results/exp_baseline_matrix/v42_pbmc_donor_celltypist_full_training/results_summary.csv",
    ROOT / "results/exp_baseline_matrix/v40_pbmc_donor_full_training_xgb_centroid_gnb/results_summary.csv",
    ROOT / "results/exp_baseline_matrix/v40_pbmc_donor_full_training_ridge/results_summary.csv",
]
OUT_DIR = (
    ROOT
    / "manufacture/mosaic_n_bioinformatics_manuscript_v1"
    / "oup-authoring-template/tables/v37"
)
RESULT_TABLE = ROOT / "results/tables" / f"mosaic_n_v37_strong_ml_expanded_baselines_{DATE}.csv"
OUTPUT_TABLE = ROOT / "output/tables" / f"mosaic_n_v37_strong_ml_expanded_baselines_{DATE}.csv"


METHOD_ORDER = [
    "CellTypist",
    "Early-fusion XGBoost",
    "RNA XGBoost",
    "ADT XGBoost",
    "Early-fusion ridge",
    "RNA ridge",
    "ADT ridge",
    "Early-fusion centroid",
    "RNA centroid",
    "ADT centroid",
]
MODALITY = {
    "CellTypist": "RNA-only",
    "RNA": "RNA-only",
    "ADT": "ADT-only",
    "Early-fusion": "RNA+ADT",
}
DEFAULT_PROTOCOL_CAVEAT = "V33 nested leave-one-donor-out; full training donors; full held-out donor test; no test-label tuning."
PROTOCOL_CAVEATS = {
    "CellTypist": (
        "V33 nested leave-one-donor-out; published CellTypist package; RNA-only; "
        "full training donors; full held-out donor test; direct predictor on train-only standardized RNA tensors."
    ),
}


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def f4(value: str) -> str:
    return f"{float(value):.4f}"


def pm(row: dict[str, str]) -> str:
    return f"{f4(row['mean'])} $\\pm$ {f4(row['ci95_margin'])}"


def esc(value: str) -> str:
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
    text = str(value)
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def load_summary() -> dict[tuple[str, str], dict[str, str]]:
    rows: dict[tuple[str, str], dict[str, str]] = {}
    for source in SOURCES:
        for row in read_rows(source):
            rows[(row["method"], row["metric"])] = row
    return rows


def method_modality(method: str) -> str:
    prefix = method.split(" ", 1)[0]
    return MODALITY.get(method, MODALITY.get(prefix, "RNA+ADT"))


def build_rows(summary: dict[tuple[str, str], dict[str, str]]) -> list[dict[str, str]]:
    output = []
    for method in METHOD_ORDER:
        if (method, "accuracy") not in summary:
            continue
        output.append(
            {
                "method": method,
                "modality": method_modality(method),
                "n_donors": summary[(method, "accuracy")]["n_donors"],
                "accuracy": pm(summary[(method, "accuracy")]),
                "weighted_f1": pm(summary[(method, "weighted_f1")]),
                "macro_f1": pm(summary[(method, "macro_f1")]),
                "balanced_accuracy": pm(summary[(method, "balanced_accuracy")]),
                "worst_donor_accuracy": f4(summary[(method, "accuracy")]["worst_donor_value"]),
                "best_donor_accuracy": f4(summary[(method, "accuracy")]["best_donor_value"]),
                "protocol_caveat": PROTOCOL_CAVEATS.get(method, DEFAULT_PROTOCOL_CAVEAT),
            }
        )
    return output


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_tex(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Method & Modality & ACC & W-F1 & M-F1 & B-ACC & Worst ACC & Best ACC \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    esc(row["method"]),
                    esc(row["modality"]),
                    row["accuracy"],
                    row["weighted_f1"],
                    row["macro_f1"],
                    row["balanced_accuracy"],
                    row["worst_donor_accuracy"],
                    row["best_donor_accuracy"],
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows = build_rows(load_summary())
    if not rows:
        raise RuntimeError("no full-training strong baseline rows found")
    write_csv(rows, RESULT_TABLE)
    write_csv(rows, OUTPUT_TABLE)
    write_csv(rows, OUT_DIR / f"strong_ml_expanded_baselines_{DATE}.csv")
    write_tex(rows, OUT_DIR / "supplement_strong_ml_expanded_baselines.tex")
    print(f"wrote {len(rows)} full-training baseline rows")


if __name__ == "__main__":
    main()
