#!/usr/bin/env python
"""Audit machine, author and external-release readiness for MOSAIC-N V33."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATE = "2026-07-23"
MANUSCRIPT_RELATIVE = Path(
    "manufacture/mosaic_n_bioinformatics_manuscript_v1/oup-authoring-template"
)
DEFAULT_RELEASE_RELATIVE = Path("output/release/mosaic_n_v33")

EXPERIMENT_EVIDENCE = [
    "results/tables/mosaic_n_v33_preflight_2026-07-23.csv",
    "results/tables/mosaic_n_v33_nested_lodo_cache_summary_2026-07-23.csv",
    "results/tables/mosaic_n_v33_donor_method_summary_2026-07-23.csv",
    "results/tables/mosaic_n_v33_paired_donor_statistics_2026-07-23.csv",
    "results/tables/mosaic_n_v33_checkpoint_ablation_metrics_2026-07-23.csv",
    "results/tables/mosaic_n_v33_panel_robustness_metrics_2026-07-23.csv",
    "results/tables/mosaic_n_v33_panel_robustness_slopes_2026-07-23.csv",
    "results/tables/mosaic_n_v33_unknown_reject_metrics_2026-07-23.csv",
    "results/tables/mosaic_n_v33_unknown_reject_summary_2026-07-23.csv",
    "results/tables/mosaic_n_v33_pdc101_mmochi_comparison_2026-07-23.csv",
    "results/tables/mosaic_n_v33_donor_attribution_stability_2026-07-23.csv",
    "results/tables/mosaic_n_v33_marker_enrichment_2026-07-23.csv",
    "results/tables/mosaic_n_v33_pdc101_mmochi_per_class_2026-07-23.csv",
    "results/tables/mosaic_n_v33_donor_per_class_summary_2026-07-23.csv",
]

AUTHOR_PLACEHOLDER_PATTERNS = [
    r"\[AUTHOR ACTION[^\]]*\]",
    r"must be supplied by the authors",
    r"must be confirmed by all authors",
    r"TODO_AUTHOR",
    r"AUTHOR_NAME",
]
EXTERNAL_PLACEHOLDER_PATTERNS = [
    r"github\.com/OWNER/REPO",
    r"zenodo\.TBD",
    r"will be supplied before submission",
    r"TODO_RELEASE",
]
LATEX_FATAL_PATTERNS = [
    "LaTeX Error",
    "Fatal error",
    "undefined references",
    "undefined citations",
    "Emergency stop",
]
EXPECTED_DONORS = {f"P{index}" for index in range(1, 9)}
EXPECTED_SEEDS = {41, 42, 43}
EXPECTED_RETRAINED_METHODS = {
    "mlp",
    "mosaic_full",
    "mosaic_no_hsr",
    "mosaic_no_kd",
}
EXPECTED_VARIANTS = {
    "rna_branch",
    "adt_branch",
    "fusion_branch",
    "uniform_fusion",
    "margin_gate",
    "margin_gate_hsr",
}
EXPECTED_PANEL_SCENARIOS = {
    "full",
    "random_10",
    "random_30",
    "random_50",
    "random_70",
    "marker_memory",
    "marker_tcell",
    "rna_only",
}
EXPECTED_UNKNOWN_TARGETS = {
    "gdT_2",
    "NK_3",
    "CD4 TCM_1",
    "CD8 TEM_4",
    "B naive lambda",
}
EXPECTED_UNKNOWN_SCORES = {
    "one_minus_max_probability",
    "one_minus_margin",
    "energy",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _row(
    check: str,
    category: str,
    responsibility: str,
    status: str,
    evidence: str,
    consequence: str,
) -> Dict[str, str]:
    return {
        "check": check,
        "category": category,
        "responsibility": responsibility,
        "status": status,
        "evidence": evidence,
        "consequence": consequence,
    }


def _nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _missing_nonempty(root: Path, paths: Sequence[str]) -> List[str]:
    return [value for value in paths if not _nonempty(root / value)]


def _read_csv(root: Path, relative_path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(root / relative_path)
    except Exception as exc:
        raise ValueError(f"{relative_path}: unreadable CSV ({exc})") from exc


def validate_experiment_evidence(root: Path) -> List[str]:
    issues = []
    try:
        preflight = _read_csv(
            root,
            "results/tables/mosaic_n_v33_preflight_2026-07-23.csv",
        )
        if len(preflight) < 10 or set(preflight["status"].astype(str)) != {"pass"}:
            issues.append("preflight is not at least 10 all-pass checks")

        caches = _read_csv(
            root,
            "results/tables/mosaic_n_v33_nested_lodo_cache_summary_2026-07-23.csv",
        )
        if set(caches["test_donor"].astype(str)) != EXPECTED_DONORS:
            issues.append("nested LODO cache summary does not cover P1-P8")
        if not caches["feature_selection_fit"].astype(str).eq(
            "training donors only"
        ).all() or not caches["scalers_fit"].astype(str).eq(
            "training donors only"
        ).all():
            issues.append("nested LODO preprocessing is not train-only")

        method_summary = _read_csv(
            root,
            "results/tables/mosaic_n_v33_donor_method_summary_2026-07-23.csv",
        )
        retrained = method_summary[
            method_summary["method"].astype(str).isin(EXPECTED_RETRAINED_METHODS)
        ]
        if set(retrained["method"].astype(str)) != EXPECTED_RETRAINED_METHODS:
            issues.append("donor summary is missing a retrained method")
        if retrained.empty or not retrained["n_donors"].astype(int).eq(8).all():
            issues.append("retrained method summary is not based on eight donors")

        paired = _read_csv(
            root,
            "results/tables/mosaic_n_v33_paired_donor_statistics_2026-07-23.csv",
        )
        if paired.empty or not paired["n_donors"].astype(int).eq(8).all():
            issues.append("paired donor statistics are incomplete")
        donor_per_class = _read_csv(
            root,
            "results/tables/mosaic_n_v33_donor_per_class_summary_2026-07-23.csv",
        )
        required_per_class_methods = donor_per_class[
            donor_per_class["method"].astype(str).isin({"mlp", "mosaic_full"})
        ]
        counts = required_per_class_methods.groupby("method")[
            "class_label"
        ].nunique()
        if (
            set(counts.index.astype(str)) != {"mlp", "mosaic_full"}
            or not counts.eq(58).all()
            or not donor_per_class["n_observed_donors"]
            .astype(int)
            .between(1, 8)
            .all()
        ):
            issues.append("donor per-class summary lacks 58 classes for MLP/full")

        ablation = _read_csv(
            root,
            "results/tables/mosaic_n_v33_checkpoint_ablation_metrics_2026-07-23.csv",
        )
        ablation_keys = ablation[
            ["test_donor", "seed", "variant"]
        ].drop_duplicates()
        if (
            set(ablation_keys["test_donor"].astype(str)) != EXPECTED_DONORS
            or set(ablation_keys["seed"].astype(int)) != EXPECTED_SEEDS
            or set(ablation_keys["variant"].astype(str)) != EXPECTED_VARIANTS
            or len(ablation_keys) != 8 * 3 * 6
        ):
            issues.append("checkpoint ablation is not the complete 8x3x6 matrix")

        panel = _read_csv(
            root,
            "results/tables/mosaic_n_v33_panel_robustness_metrics_2026-07-23.csv",
        )
        panel_keys = panel[["test_donor", "seed", "scenario"]].drop_duplicates()
        if (
            set(panel_keys["test_donor"].astype(str)) != EXPECTED_DONORS
            or set(panel_keys["seed"].astype(int)) != EXPECTED_SEEDS
            or set(panel_keys["scenario"].astype(str))
            != EXPECTED_PANEL_SCENARIOS
            or len(panel_keys) != 8 * 3 * 8
        ):
            issues.append("panel robustness is not the complete 8x3x8 matrix")

        unknown = _read_csv(
            root,
            "results/tables/mosaic_n_v33_unknown_reject_metrics_2026-07-23.csv",
        )
        unknown_keys = unknown[
            [
                "target_label",
                "seed",
                "score",
                "known_coverage_target",
            ]
        ].drop_duplicates()
        if (
            set(unknown_keys["target_label"].astype(str))
            != EXPECTED_UNKNOWN_TARGETS
            or set(unknown_keys["seed"].astype(int)) != EXPECTED_SEEDS
            or set(unknown_keys["score"].astype(str))
            != EXPECTED_UNKNOWN_SCORES
            or set(
                np.round(
                    unknown_keys["known_coverage_target"].astype(float),
                    2,
                )
            )
            != {0.80, 0.95}
            or len(unknown_keys) != 5 * 3 * 3 * 2
        ):
            issues.append("unknown/reject evidence is not the complete 5x3x3x2 matrix")

        pdc = _read_csv(
            root,
            "results/tables/mosaic_n_v33_pdc101_mmochi_comparison_2026-07-23.csv",
        )
        if (
            set(pdc["method"].astype(str)) != {"MOSAIC-N", "MMoCHi"}
            or set(pdc["metric"].astype(str))
            != {"accuracy", "weighted_f1", "macro_f1"}
            or len(pdc[["method", "metric"]].drop_duplicates()) != 6
        ):
            issues.append("PDC101 comparison is missing method-metric rows")
        pdc_per_class = _read_csv(
            root,
            "results/tables/mosaic_n_v33_pdc101_mmochi_per_class_2026-07-23.csv",
        )
        pdc_class_keys = pdc_per_class[
            ["method", "class_label"]
        ].drop_duplicates()
        if (
            set(pdc_class_keys["method"].astype(str))
            != {"MOSAIC-N", "MMoCHi"}
            or pdc_class_keys["class_label"].astype(str).nunique() != 8
            or len(pdc_class_keys) != 16
            or not pdc_per_class["support"].astype(int).gt(0).all()
        ):
            issues.append("PDC101 per-class comparison is not two methods x eight classes")

        attribution = _read_csv(
            root,
            "results/tables/mosaic_n_v33_donor_attribution_stability_2026-07-23.csv",
        )
        attribution_donors = set(attribution["donor_a"].astype(str)) | set(
            attribution["donor_b"].astype(str)
        )
        if (
            attribution_donors != EXPECTED_DONORS
            or set(attribution["seed"].astype(int)) != EXPECTED_SEEDS
            or set(attribution["modality"].astype(str)) != {"RNA", "ADT"}
            or attribution["class_label"].astype(str).nunique() != 6
        ):
            issues.append("attribution stability lacks donor, seed, class or modality coverage")

        markers = _read_csv(
            root,
            "results/tables/mosaic_n_v33_marker_enrichment_2026-07-23.csv",
        )
        marker_keys = markers[["class_label", "modality"]].drop_duplicates()
        if (
            marker_keys["class_label"].astype(str).nunique() != 6
            or set(marker_keys["modality"].astype(str)) != {"RNA", "ADT"}
            or len(marker_keys) != 12
        ):
            issues.append("marker enrichment does not contain six classes x two modalities")
    except (KeyError, ValueError) as exc:
        issues.append(str(exc))
    return issues


def _verify_release_checksums(release_dir: Path) -> tuple[bool, str]:
    checksum_path = release_dir / "CHECKSUMS.sha256"
    if not _nonempty(checksum_path):
        return False, "CHECKSUMS.sha256 is missing or empty"
    checked = 0
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("  ", 1)
        if len(parts) != 2 or not re.fullmatch(r"[0-9a-f]{64}", parts[0]):
            return False, f"malformed checksum line: {line}"
        payload = release_dir / parts[1]
        if not _nonempty(payload):
            return False, f"checksummed file is missing or empty: {parts[1]}"
        if _sha256(payload) != parts[0]:
            return False, f"checksum mismatch: {parts[1]}"
        checked += 1
    if checked == 0:
        return False, "CHECKSUMS.sha256 contains no payload entries"
    return True, f"{checked} packaged files passed SHA-256 verification"


def audit_submission_readiness(
    root: Path = ROOT,
    release_dir: Optional[Path] = None,
) -> pd.DataFrame:
    root = root.resolve()
    manuscript = root / MANUSCRIPT_RELATIVE
    release = (
        root / DEFAULT_RELEASE_RELATIVE
        if release_dir is None
        else release_dir.resolve()
    )
    main_path = manuscript / "main.tex"
    supplement_path = manuscript / "supplementary.tex"
    main_text = (
        main_path.read_text(encoding="utf-8", errors="replace")
        if main_path.is_file()
        else ""
    )
    supplement_text = (
        supplement_path.read_text(encoding="utf-8", errors="replace")
        if supplement_path.is_file()
        else ""
    )
    combined_text = main_text + "\n" + supplement_text

    missing_experiments = _missing_nonempty(root, EXPERIMENT_EVIDENCE)
    experiment_structure_issues = (
        []
        if missing_experiments
        else validate_experiment_evidence(root)
    )
    table_paths = sorted((manuscript / "tables/v33").glob("*.tex"))
    valid_tables = [path for path in table_paths if _nonempty(path)]
    figure_paths = sorted(
        path
        for path in (root / "output/figures").glob("mosaic_n_v33_evidence*.*")
        if path.suffix.lower() in {".pdf", ".png", ".svg"} and _nonempty(path)
    )

    latex_required = [
        manuscript / "main.tex",
        manuscript / "supplementary.tex",
        manuscript / "main.pdf",
        manuscript / "supplementary.pdf",
        manuscript / "main.log",
        manuscript / "supplementary.log",
    ]
    missing_latex = [str(path.relative_to(root)) for path in latex_required if not _nonempty(path)]
    fatal_messages = []
    for log_name in ["main.log", "supplementary.log"]:
        path = manuscript / log_name
        text = (
            path.read_text(encoding="utf-8", errors="replace")
            if path.is_file()
            else ""
        )
        for pattern in LATEX_FATAL_PATTERNS:
            if pattern.lower() in text.lower():
                fatal_messages.append(f"{log_name}: {pattern}")

    release_required = [
        release / "release_manifest.csv",
        release / "commands.txt",
        release / "environment_metadata.json",
        release / "CHECKSUMS.sha256",
    ]
    missing_release = [
        path.name for path in release_required if not _nonempty(path)
    ]
    checksum_ok, checksum_evidence = (
        _verify_release_checksums(release)
        if not missing_release
        else (False, "release metadata is incomplete")
    )
    release_smoke_fixture = (
        release
        / "testdata/experiments/reproducibility/testdata/v33_synthetic_smoke.csv"
    )
    smoke_fixture_ok = _nonempty(release_smoke_fixture)
    commands_text = (
        (release / "commands.txt").read_text(
            encoding="utf-8",
            errors="replace",
        )
        if _nonempty(release / "commands.txt")
        else ""
    )
    smoke_command_ok = (
        "v33_release_smoke_test.py" in commands_text
        and "--release-dir" in commands_text
    )
    release_integrity_ok = (
        not missing_release
        and checksum_ok
        and smoke_fixture_ok
        and smoke_command_ok
    )
    if not smoke_fixture_ok:
        release_evidence = (
            "The synthetic smoke-test data file is missing or empty: "
            + str(release_smoke_fixture.relative_to(root))
        )
    elif not smoke_command_ok:
        release_evidence = (
            "commands.txt does not contain the packaged synthetic smoke-test command."
        )
    elif missing_release:
        release_evidence = (
            "Missing or empty release files: " + ", ".join(missing_release)
        )
    else:
        release_evidence = checksum_evidence

    author_placeholders = [
        pattern
        for pattern in AUTHOR_PLACEHOLDER_PATTERNS
        if re.search(pattern, combined_text, flags=re.IGNORECASE)
    ]
    external_placeholders = [
        pattern
        for pattern in EXTERNAL_PLACEHOLDER_PATTERNS
        if re.search(pattern, combined_text, flags=re.IGNORECASE)
    ]
    public_repository = re.search(
        r"https://github\.com/(?!OWNER/REPO)([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)",
        combined_text,
        flags=re.IGNORECASE,
    )
    archive_doi = re.search(
        r"10\.5281/zenodo\.[0-9]+",
        combined_text,
        flags=re.IGNORECASE,
    )
    license_present = any(
        _nonempty(release / name) for name in ["LICENSE", "LICENSE.txt", "COPYING"]
    )
    review_confirmation = release / "AUTHOR_REVIEW_CONFIRMATION.txt"
    review_text = (
        review_confirmation.read_text(encoding="utf-8", errors="replace")
        if review_confirmation.is_file()
        else ""
    )
    human_review_ok = bool(re.search(r"confirmed_by\s*=\s*\S+", review_text))

    rows = [
        _row(
            "experiment_evidence_complete",
            "experiments",
            "machine_closed",
            "pass"
            if not missing_experiments and not experiment_structure_issues
            else "blocker",
            (
                f"All {len(EXPERIMENT_EVIDENCE)} prespecified compact evidence files exist."
                if not missing_experiments and not experiment_structure_issues
                else (
                    "Missing or empty: " + ", ".join(missing_experiments)
                    if missing_experiments
                    else "Structural issues: "
                    + "; ".join(experiment_structure_issues)
                )
            ),
            "Missing experimental evidence prevents final claims and table regeneration.",
        ),
        _row(
            "manuscript_tables_complete",
            "tables_figures",
            "machine_closed",
            "pass" if len(valid_tables) >= 4 else "blocker",
            f"Found {len(valid_tables)} non-empty V33 LaTeX tables; at least 4 are required.",
            "The V33 donor, robustness, reject and attribution evidence must be represented.",
        ),
        _row(
            "manuscript_figures_complete",
            "tables_figures",
            "machine_closed",
            "pass" if figure_paths else "blocker",
            (
                "Evidence figure(s): "
                + ", ".join(str(path.relative_to(root)) for path in figure_paths)
                if figure_paths
                else "No non-empty output/figures/mosaic_n_v33_evidence.{pdf,png,svg} found."
            ),
            "At least one regenerated V33 evidence figure is required.",
        ),
        _row(
            "latex_sources_and_build",
            "latex",
            "machine_closed",
            "pass" if not missing_latex and not fatal_messages else "blocker",
            (
                "Both LaTeX sources and PDFs exist; logs contain no fatal, citation or reference errors."
                if not missing_latex and not fatal_messages
                else "Missing: "
                + ", ".join(missing_latex)
                + "; log blockers: "
                + ", ".join(fatal_messages)
            ),
            "A non-compiling or unresolved manuscript package cannot be submitted.",
        ),
        _row(
            "release_candidate_integrity",
            "reproducibility",
            "machine_closed",
            "pass" if release_integrity_ok else "blocker",
            release_evidence,
            "The compact code/config/manifest package must be checksummed and reproducible.",
        ),
        _row(
            "author_metadata_finalized",
            "placeholder_information",
            "author_action",
            "pass" if main_text and not author_placeholders else "blocker",
            (
                "No author-action placeholder pattern remains."
                if main_text and not author_placeholders
                else "Author placeholders: " + ", ".join(author_placeholders or ["main.tex missing"])
            ),
            "Names, affiliations, funding and CRediT roles require author confirmation.",
        ),
        _row(
            "human_authorship_review",
            "placeholder_information",
            "author_action",
            "pass" if human_review_ok else "blocker",
            (
                "AUTHOR_REVIEW_CONFIRMATION.txt records a named confirmer."
                if human_review_ok
                else "A named human review confirmation has not been recorded in the external release package."
            ),
            "Final scientific claims and journal-policy compliance require accountable human review.",
        ),
        _row(
            "public_repository_release",
            "placeholder_information",
            "external_release",
            "pass"
            if public_repository and license_present and not external_placeholders
            else "blocker",
            (
                f"Public repository {public_repository.group(0)} and release license are present."
                if public_repository and license_present and not external_placeholders
                else "A real public repository URL, release license and removal of release placeholders are required."
            ),
            "Creating and publishing the repository requires an external account.",
        ),
        _row(
            "archival_doi_release",
            "placeholder_information",
            "external_release",
            "pass" if archive_doi and not external_placeholders else "blocker",
            (
                f"Archive DOI found: {archive_doi.group(0)}."
                if archive_doi and not external_placeholders
                else "No final numeric Zenodo DOI is present, or an external-release placeholder remains."
            ),
            "Minting an archival DOI requires an external repository release.",
        ),
    ]
    return pd.DataFrame(rows)


def _render_markdown(frame: pd.DataFrame) -> str:
    lines = [
        "# MOSAIC-N V33 submission readiness audit",
        "",
        f"Date: {DATE}",
        "",
        "| Check | Category | Responsibility | Status | Evidence | Consequence |",
        "|---|---|---|---|---|---|",
    ]
    for row in frame.itertuples(index=False):
        values = [
            row.check,
            row.category,
            row.responsibility,
            row.status,
            row.evidence,
            row.consequence,
        ]
        lines.append(
            "| "
            + " | ".join(str(value).replace("|", "/") for value in values)
            + " |"
        )
    machine = frame.loc[frame["responsibility"] == "machine_closed", "status"]
    lines.extend(
        [
            "",
            "## Decision",
            "",
            (
                "Machine-closed evidence is complete."
                if machine.eq("pass").all()
                else "Machine-closed evidence remains blocked."
            ),
            "Submission is ready only when every row is pass.",
            "",
        ]
    )
    return "\n".join(lines)


def write_audit(frame: pd.DataFrame, root: Path = ROOT) -> None:
    payload = {
        "date": DATE,
        "machine_ready": bool(
            frame.loc[
                frame["responsibility"] == "machine_closed", "status"
            ].eq("pass").all()
        ),
        "submission_ready": bool(frame["status"].eq("pass").all()),
        "status_counts": frame["status"].value_counts().to_dict(),
        "responsibility_status_counts": (
            frame.groupby(["responsibility", "status"]).size().to_dict()
        ),
    }
    payload["responsibility_status_counts"] = {
        f"{key[0]}:{key[1]}": value
        for key, value in payload["responsibility_status_counts"].items()
    }
    for base in [root / "results/tables", root / "output/tables"]:
        base.mkdir(parents=True, exist_ok=True)
        stem = f"mosaic_n_v33_submission_readiness_audit_{DATE}"
        frame.to_csv(base / f"{stem}.csv", index=False)
        (base / f"{stem}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (base / f"{stem}.md").write_text(
            _render_markdown(frame),
            encoding="utf-8",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--release-dir", type=Path)
    args = parser.parse_args()
    frame = audit_submission_readiness(args.root, args.release_dir)
    write_audit(frame, args.root.resolve())
    counts = frame.groupby(["responsibility", "status"]).size().to_dict()
    print(f"V33 submission readiness: {counts}")
    machine_blockers = frame[
        (frame["responsibility"] == "machine_closed")
        & (frame["status"] == "blocker")
    ]
    if not machine_blockers.empty:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
