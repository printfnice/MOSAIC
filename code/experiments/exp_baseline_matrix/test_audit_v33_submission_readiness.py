import hashlib
import itertools
from pathlib import Path

import pandas as pd

from experiments.exp_baseline_matrix.audit_v33_submission_readiness import (
    EXPERIMENT_EVIDENCE,
    audit_submission_readiness,
)


def _write(path: Path, text: str = "fixture\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _write_experiment_evidence(root: Path) -> None:
    donors = [f"P{index}" for index in range(1, 9)]
    seeds = [41, 42, 43]
    for relative_path in EXPERIMENT_EVIDENCE:
        _write(root / relative_path, "metric,value\nfixture,1\n")
    _write_frame(
        root / EXPERIMENT_EVIDENCE[0],
        pd.DataFrame({"check": [f"c{i}" for i in range(10)], "status": ["pass"] * 10}),
    )
    _write_frame(
        root / EXPERIMENT_EVIDENCE[1],
        pd.DataFrame(
            {
                "test_donor": donors,
                "feature_selection_fit": ["training donors only"] * 8,
                "scalers_fit": ["training donors only"] * 8,
            }
        ),
    )
    _write_frame(
        root / EXPERIMENT_EVIDENCE[2],
        pd.DataFrame(
            {
                "method": ["mlp", "mosaic_full", "mosaic_no_hsr", "mosaic_no_kd"],
                "n_donors": [8] * 4,
            }
        ),
    )
    _write_frame(
        root / EXPERIMENT_EVIDENCE[3],
        pd.DataFrame({"reference": ["mosaic_full"], "n_donors": [8]}),
    )
    _write_frame(
        root / EXPERIMENT_EVIDENCE[4],
        pd.DataFrame(
            [
                {"test_donor": donor, "seed": seed, "variant": variant}
                for donor, seed, variant in itertools.product(
                    donors,
                    seeds,
                    [
                        "rna_branch",
                        "adt_branch",
                        "fusion_branch",
                        "uniform_fusion",
                        "margin_gate",
                        "margin_gate_hsr",
                    ],
                )
            ]
        ),
    )
    _write_frame(
        root / EXPERIMENT_EVIDENCE[5],
        pd.DataFrame(
            [
                {"test_donor": donor, "seed": seed, "scenario": scenario}
                for donor, seed, scenario in itertools.product(
                    donors,
                    seeds,
                    [
                        "full",
                        "random_10",
                        "random_30",
                        "random_50",
                        "random_70",
                        "marker_memory",
                        "marker_tcell",
                        "rna_only",
                    ],
                )
            ]
        ),
    )
    _write_frame(
        root / EXPERIMENT_EVIDENCE[7],
        pd.DataFrame(
            [
                {
                    "target_label": target,
                    "seed": seed,
                    "score": score,
                    "known_coverage_target": coverage,
                }
                for target, seed, score, coverage in itertools.product(
                    [
                        "gdT_2",
                        "NK_3",
                        "CD4 TCM_1",
                        "CD8 TEM_4",
                        "B naive lambda",
                    ],
                    seeds,
                    [
                        "one_minus_max_probability",
                        "one_minus_margin",
                        "energy",
                    ],
                    [0.95, 0.80],
                )
            ]
        ),
    )
    _write_frame(
        root / EXPERIMENT_EVIDENCE[9],
        pd.DataFrame(
            [
                {"method": method, "metric": metric}
                for method, metric in itertools.product(
                    ["MOSAIC-N", "MMoCHi"],
                    ["accuracy", "weighted_f1", "macro_f1"],
                )
            ]
        ),
    )
    _write_frame(
        root / EXPERIMENT_EVIDENCE[10],
        pd.DataFrame(
            [
                {
                    "donor_a": "P1",
                    "donor_b": donor_b,
                    "seed": seed,
                    "class_label": class_label,
                    "modality": modality,
                }
                for donor_b, seed, class_label, modality in itertools.product(
                    donors[1:],
                    seeds,
                    [f"class_{index}" for index in range(6)],
                    ["RNA", "ADT"],
                )
            ]
        ),
    )
    _write_frame(
        root / EXPERIMENT_EVIDENCE[11],
        pd.DataFrame(
            [
                {"class_label": class_label, "modality": modality}
                for class_label, modality in itertools.product(
                    [f"class_{index}" for index in range(6)],
                    ["RNA", "ADT"],
                )
            ]
        ),
    )
    _write_frame(
        root / EXPERIMENT_EVIDENCE[12],
        pd.DataFrame(
            [
                {
                    "method": method,
                    "class_label": f"class_{class_index}",
                    "support": 10,
                }
                for method, class_index in itertools.product(
                    ["MOSAIC-N", "MMoCHi"],
                    range(8),
                )
            ]
        ),
    )
    _write_frame(
        root / EXPERIMENT_EVIDENCE[13],
        pd.DataFrame(
            [
                {
                    "method": method,
                    "class_label": f"class_{class_index}",
                    "n_observed_donors": 8,
                }
                for method, class_index in itertools.product(
                    ["mlp", "mosaic_full"],
                    range(58),
                )
            ]
        ),
    )


def _machine_complete_fixture(tmp_path: Path, placeholders: bool = True) -> Path:
    root = tmp_path / "repo"
    _write_experiment_evidence(root)

    manuscript = (
        root
        / "manufacture/mosaic_n_bioinformatics_manuscript_v1/oup-authoring-template"
    )
    for index in range(4):
        _write(
            manuscript / f"tables/v33/table_{index}.tex",
            "\\begin{tabular}{cc}A&B\\\\\\end{tabular}\n",
        )
    _write(root / "output/figures/mosaic_n_v33_evidence.png", "png fixture")
    _write(manuscript / "main.pdf", "pdf fixture")
    _write(manuscript / "supplementary.pdf", "pdf fixture")
    _write(manuscript / "main.log", "Output written on main.pdf")
    _write(manuscript / "supplementary.log", "Output written on supplementary.pdf")

    if placeholders:
        main_text = r"""
\author{[AUTHOR ACTION: confirm authors]}
\section{Data availability}
Code will be supplied before submission at
\url{https://github.com/OWNER/REPO}; archive DOI: 10.5281/zenodo.TBD.
"""
    else:
        main_text = r"""
\author{Lin Hao Wu}
\section{Data availability}
Code: \url{https://github.com/example-lab/mosaic-n}.
Archive: \url{https://doi.org/10.5281/zenodo.1234567}.
"""
    _write(manuscript / "main.tex", main_text)
    _write(manuscript / "supplementary.tex", "\\section{Supplement}\n")

    release = root / "output/release/mosaic_n_v33"
    _write(release / "release_manifest.csv", "category,packaged_path\ncode,code/a.py\n")
    _write(
        release / "commands.txt",
        "python code/experiments/reproducibility/v33_release_smoke_test.py "
        "--release-dir .\n",
    )
    _write(release / "environment_metadata.json", '{"python_version":"3"}\n')
    _write(
        release
        / "testdata/experiments/reproducibility/testdata/v33_synthetic_smoke.csv",
        "cell_id,donor,split,label,rna_a,adt_a\n",
    )
    checksum_lines = []
    for path in [
        release / "commands.txt",
        release / "environment_metadata.json",
        release / "release_manifest.csv",
        release
        / "testdata/experiments/reproducibility/testdata/v33_synthetic_smoke.csv",
    ]:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        checksum_lines.append(f"{digest}  {path.relative_to(release)}")
    _write(release / "CHECKSUMS.sha256", "\n".join(checksum_lines) + "\n")
    if not placeholders:
        _write(release / "LICENSE", "BSD 3-Clause License\n")
    return root


def test_audit_distinguishes_responsibility_and_machine_categories(
    tmp_path: Path,
) -> None:
    root = _machine_complete_fixture(tmp_path, placeholders=True)

    audit = audit_submission_readiness(root)
    indexed = audit.set_index("check")

    assert set(audit["responsibility"]) == {
        "machine_closed",
        "author_action",
        "external_release",
    }
    assert set(audit["status"]) <= {"pass", "blocker"}
    for check in [
        "experiment_evidence_complete",
        "manuscript_tables_complete",
        "manuscript_figures_complete",
        "latex_sources_and_build",
        "release_candidate_integrity",
    ]:
        assert indexed.loc[check, "responsibility"] == "machine_closed"
        assert indexed.loc[check, "status"] == "pass"
    assert indexed.loc["author_metadata_finalized", "status"] == "blocker"
    assert indexed.loc["public_repository_release", "status"] == "blocker"
    assert indexed.loc["archival_doi_release", "status"] == "blocker"


def test_missing_experiment_or_latex_error_is_a_machine_blocker(
    tmp_path: Path,
) -> None:
    root = _machine_complete_fixture(tmp_path, placeholders=False)
    (root / EXPERIMENT_EVIDENCE[0]).unlink()
    manuscript = (
        root
        / "manufacture/mosaic_n_bioinformatics_manuscript_v1/oup-authoring-template"
    )
    _write(manuscript / "main.log", "! LaTeX Error: fixture failure\n")

    audit = audit_submission_readiness(root).set_index("check")

    assert audit.loc["experiment_evidence_complete", "status"] == "blocker"
    assert EXPERIMENT_EVIDENCE[0] in audit.loc[
        "experiment_evidence_complete", "evidence"
    ]
    assert audit.loc["latex_sources_and_build", "status"] == "blocker"
    assert audit.loc["latex_sources_and_build", "responsibility"] == "machine_closed"


def test_incomplete_experiment_matrix_is_a_machine_blocker(
    tmp_path: Path,
) -> None:
    root = _machine_complete_fixture(tmp_path, placeholders=False)
    ablation_path = root / EXPERIMENT_EVIDENCE[4]
    ablation = pd.read_csv(ablation_path)
    ablation = ablation[~ablation["test_donor"].eq("P8")]
    ablation.to_csv(ablation_path, index=False)

    audit = audit_submission_readiness(root).set_index("check")

    assert audit.loc["experiment_evidence_complete", "status"] == "blocker"
    assert "8x3x6" in audit.loc["experiment_evidence_complete", "evidence"]


def test_resolved_metadata_and_public_release_actions_can_pass(
    tmp_path: Path,
) -> None:
    root = _machine_complete_fixture(tmp_path, placeholders=False)

    audit = audit_submission_readiness(root).set_index("check")

    assert audit.loc["author_metadata_finalized", "status"] == "pass"
    assert audit.loc["public_repository_release", "status"] == "pass"
    assert audit.loc["archival_doi_release", "status"] == "pass"


def test_missing_synthetic_testdata_is_a_machine_blocker(tmp_path: Path) -> None:
    root = _machine_complete_fixture(tmp_path, placeholders=False)
    release = root / "output/release/mosaic_n_v33"
    fixture = (
        release
        / "testdata/experiments/reproducibility/testdata/v33_synthetic_smoke.csv"
    )
    fixture.unlink()

    audit = audit_submission_readiness(root).set_index("check")

    assert audit.loc["release_candidate_integrity", "status"] == "blocker"
    assert "synthetic smoke-test data" in audit.loc[
        "release_candidate_integrity", "evidence"
    ]
