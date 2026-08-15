#!/usr/bin/env python
"""Executable v8 audit for manuscript numbering and evidence identities."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANUSCRIPT = ROOT / "manufacture/mosaic_n_bioinformatics_manuscript_v1/oup-authoring-template"
SUPPLEMENT = MANUSCRIPT / "supplementary.tex"
MAIN = MANUSCRIPT / "main.tex"
MARKER_FRAGMENT = MANUSCRIPT / "tables/v7/supplement_marker_mechanism_audit.tex"
S6_FRAGMENT = MANUSCRIPT / "tables/v45/supplement_continuous_branch_audit.tex"
SCHEMA = MANUSCRIPT / "tables/v50/main_audit_record_schema.tex"
S5_FRAGMENT = MANUSCRIPT / "tables/v34/supplement_auditability_closure.tex"

EXPECTED_LABELS = [
    "tab:splits",
    "tab:donor-full",
    "tab:nokd-mlp-paired",
    "tab:branch-supervision",
    "tab:auditability-closure",
    "tab:continuous-branch-audit",
    "tab:margin-correctness",
    "tab:panel",
    "tab:unknown-aggregate",
    "tab:unknown-target",
    "tab:unknown-decomposition",
    "tab:light-chain",
    "tab:marker-mechanism",
    "tab:pdc-aggregate",
    "tab:pdc-class",
    "tab:computational-cost",
    "tab:strong-ml-baselines",
    "tab:xgb-paired",
    "tab:published-fixed-baselines",
    "tab:clr-sensitivity",
    "tab:attribution",
    "tab:macro-pr",
    "tab:cross-dataset-breadth",
    "tab:panel-aware-training",
    "tab:rna-cp10k-sensitivity",
]


def _expand_inputs(text: str, base: Path) -> str:
    pattern = re.compile(r"\\input\{([^}]+)\}")

    def replace(match: re.Match[str]) -> str:
        candidate = base / match.group(1)
        if candidate.suffix != ".tex":
            candidate = candidate.with_suffix(".tex")
        if not candidate.is_file():
            return match.group(0)
        return _expand_inputs(candidate.read_text(encoding="utf-8"), candidate.parent)

    return pattern.sub(replace, text)


def _expand_numbers(tokens: list[str]) -> list[int]:
    values: list[int] = []
    for token in tokens:
        match = re.fullmatch(r"S(\d+)(?:--S(\d+))?", token)
        if not match:
            continue
        start = int(match.group(1))
        end = int(match.group(2) or start)
        values.extend(range(start, end + 1))
    return values


def _main_citations(text: str) -> list[int]:
    pattern = re.compile(r"Supplementary Tables?(?:~|\s+)S(\d+)(?:--S(\d+))?")
    values: list[int] = []
    for match in pattern.finditer(text):
        start = int(match.group(1))
        end = int(match.group(2) or start)
        for value in range(start, end + 1):
            if value not in values:
                values.append(value)
    return values


def audit() -> dict[str, object]:
    supplement = SUPPLEMENT.read_text(encoding="utf-8")
    expanded = _expand_inputs(supplement, MANUSCRIPT)
    main = MAIN.read_text(encoding="utf-8")
    index_tokens = re.findall(r"^((?:S\d+(?:--S\d+)?|U\d+))\s*&", supplement, flags=re.MULTILINE)
    s_tokens = [token for token in index_tokens if token.startswith("S")]
    u_tokens = [token for token in index_tokens if token.startswith("U")]
    physical = re.findall(r"\\label\{(tab:[^}]+)\}", expanded)
    citations = _main_citations(main)

    if physical != EXPECTED_LABELS:
        raise AssertionError(f"supplement physical labels drifted: {physical}")
    if _expand_numbers(s_tokens) != list(range(1, 26)):
        raise AssertionError(f"supplement index drifted: {s_tokens}")
    if u_tokens != ["U0", "U1", "U2", "U3"]:
        raise AssertionError(f"unnumbered item index drifted: {u_tokens}")
    if citations != list(range(1, 26)):
        raise AssertionError(f"main first-citation order drifted: {citations}")
    if "Audit subset (1 feature)" in S6_FRAGMENT.read_text(encoding="utf-8") or "Audit subset (4 features)" in S6_FRAGMENT.read_text(encoding="utf-8"):
        raise AssertionError("S6 still mixes canonical direct estimates with 1/4-feature subset rows")
    s5 = S5_FRAGMENT.read_text(encoding="utf-8")
    s6 = S6_FRAGMENT.read_text(encoding="utf-8")
    conflict_pattern = re.compile(
        r"^Top-label conflict.*?risk at 90\\% coverage.*?"
        r"(0\.\d+) \[(0\.\d+), (0\.\d+)\]",
        flags=re.MULTILINE,
    )
    s5_conflict = conflict_pattern.search(s5)
    s6_conflict = conflict_pattern.search(s6)
    expected_conflict = ("0.0633", "0.0605", "0.0661")
    if not s5_conflict or not s6_conflict:
        raise AssertionError("could not locate the repeated top-label-conflict risk row")
    if s5_conflict.groups() != expected_conflict or s6_conflict.groups() != expected_conflict:
        raise AssertionError(
            "S5/S6 top-label-conflict risk rows disagree with canonical v46 estimate: "
            f"S5={s5_conflict.groups()}, S6={s6_conflict.groups()}"
        )
    schema = SCHEMA.read_text(encoding="utf-8")
    if "disagreement=0.667" in schema or "branch conflict=1" not in schema:
        raise AssertionError("Table 2 branch-conflict example is not binary")
    if "fig:risk-coverage" in main or "fig:marker-support" not in main:
        raise AssertionError("main Figure 3 is still the duplicated risk--coverage plot")
    marker_text = MARKER_FRAGMENT.read_text(encoding="utf-8")
    if "tab:marker-mechanism" not in marker_text:
        raise AssertionError("marker longtable is not explicitly labelled S13")
    if "CD8 Naive & cd8\\_n & 5/5 & 5 & 0.714 & 0.9260 & 0.9734 & 0.0474" not in marker_text:
        raise AssertionError("CD8 displayed delta is not computed from the displayed F1 values")

    result = {
        "physical_table_count": len(physical),
        "physical_table_labels": physical,
        "supplement_index": s_tokens,
        "unnumbered_index": u_tokens,
        "main_first_citations": citations,
        "s6_direct_subset_identity": "pass",
        "s5_s6_conflict_identity": "pass",
        "table2_binary_conflict_identity": "pass",
        "main_figure3_distinct_from_risk_coverage": "pass",
    }
    output = ROOT / "results/tables/mosaic_n_v8_manuscript_consistency_audit_2026-08-15.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"wrote {output}")
    return result


if __name__ == "__main__":
    audit()
