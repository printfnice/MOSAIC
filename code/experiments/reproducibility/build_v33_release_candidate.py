#!/usr/bin/env python
"""Build a lightweight, checksummed V33 reproducibility release candidate."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATE = "2026-07-23"
DEFAULT_OUTPUT = ROOT / "output/release/mosaic_n_v33"
DEFAULT_MAX_FILE_BYTES = 10 * 1024 * 1024

DEFAULT_REQUIRED_ARTIFACTS = [
    "results/tables/mosaic_n_v33_preflight_2026-07-23.csv",
    "results/tables/mosaic_n_v33_nested_lodo_cache_summary_2026-07-23.csv",
    "results/tables/mosaic_n_v33_donor_method_summary_2026-07-23.csv",
    "results/tables/mosaic_n_v33_paired_donor_statistics_2026-07-23.csv",
    "results/tables/mosaic_n_v33_donor_per_class_summary_2026-07-23.csv",
    "results/tables/mosaic_n_v33_checkpoint_ablation_metrics_2026-07-23.csv",
    "results/tables/mosaic_n_v33_panel_robustness_metrics_2026-07-23.csv",
    "results/tables/mosaic_n_v33_panel_robustness_slopes_2026-07-23.csv",
    "results/tables/mosaic_n_v33_unknown_reject_metrics_2026-07-23.csv",
    "results/tables/mosaic_n_v33_unknown_reject_summary_2026-07-23.csv",
    "results/tables/mosaic_n_v33_pdc101_mmochi_comparison_2026-07-23.csv",
    "results/tables/mosaic_n_v33_pdc101_mmochi_per_class_2026-07-23.csv",
    "results/tables/mosaic_n_v33_donor_attribution_stability_2026-07-23.csv",
    "results/tables/mosaic_n_v33_marker_enrichment_2026-07-23.csv",
]

DEFAULT_COMMANDS = [
    "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 conda run -n TOSICA python experiments/reproducibility/v33_release_smoke_test.py --release-dir output/release/mosaic_n_v33",
    "conda run -n TOSICA python experiments/exp_generalization/mosaic_n_v33/audit_v33_preflight.py",
    "conda run -n TOSICA python experiments/exp_generalization/mosaic_n_v33/build_pbmc_nested_donor_cache.py --all",
    "conda run -n TOSICA python experiments/exp_generalization/mosaic_n_v33/run_v33_donor_matrix.py --all",
    "conda run -n TOSICA python experiments/exp_generalization/mosaic_n_v33/evaluate_v33_checkpoint_ablations.py --require-complete",
    "conda run -n TOSICA python experiments/exp_generalization/mosaic_n_v33/build_v33_donor_summary.py --require-complete",
    "conda run -n TOSICA python experiments/exp_missing_modality/mosaic_n_v33/evaluate_v33_panel_robustness.py --require-complete",
    "conda run -n TOSICA python experiments/exp_unknown_celltype/mosaic_n_v33/build_v33_unknown_protocol.py --all",
    "conda run -n TOSICA python experiments/exp_unknown_celltype/mosaic_n_v33/run_v33_unknown_matrix.py --all",
    "conda run -n TOSICA python experiments/exp_unknown_celltype/mosaic_n_v33/evaluate_v33_unknown_reject.py --require-complete",
    "conda run -n MMoCHi python experiments/exp_generalization/mosaic_n_v33/run_v33_pdc101_mosaic_n.py",
    "conda run -n TOSICA python experiments/exp_generalization/mosaic_n_v33/build_v33_pdc101_comparison.py",
    "conda run -n TOSICA python experiments/exp_explainability/mosaic_n_v33/analyze_v33_donor_attribution.py --require-complete",
]

FORBIDDEN_SUFFIXES = {
    ".ckpt",
    ".h5",
    ".h5ad",
    ".npy",
    ".npz",
    ".pkl",
    ".pt",
    ".pth",
    ".tar",
    ".tgz",
    ".zip",
}
FORBIDDEN_TOP_LEVEL = {"cache", "data"}


class ReleaseBlockedError(RuntimeError):
    """Raised when a release candidate cannot be built without missing evidence."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_relative_path(root: Path, value: str) -> Path:
    candidate = Path(value)
    absolute = candidate if candidate.is_absolute() else root / candidate
    resolved = absolute.resolve()
    try:
        return resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ReleaseBlockedError(
            f"release source is outside repository root: {value}"
        ) from exc


def _is_forbidden(relative_path: Path) -> bool:
    return (
        bool(relative_path.parts)
        and relative_path.parts[0] in FORBIDDEN_TOP_LEVEL
    ) or relative_path.suffix.lower() in FORBIDDEN_SUFFIXES


def _expand_globs(root: Path, patterns: Sequence[str]) -> List[str]:
    paths = set()
    for pattern in patterns:
        for path in root.glob(pattern):
            if path.is_file():
                paths.add(path.relative_to(root).as_posix())
    return sorted(paths)


def default_source_groups(root: Path) -> Dict[str, List[str]]:
    code = _expand_globs(
        root,
        [
            "experiments/*/mosaic_n_v33/*.py",
            "experiments/exp_baseline_matrix/*v33*.py",
            "experiments/reproducibility/*v33*.py",
        ],
    )
    code.extend(
        [
            "experiments/exp_generalization/original_mosaic_strict/build_pseudo_unknown_cache.py",
            "experiments/exp_generalization/original_mosaic_strict/run_l3_strict_mosaic.py",
            "experiments/exp_generalization/original_mosaic_strict/run_mlp_strict_baseline.py",
            "experiments/exp_generalization/original_mosaic_strict/run_mosaic_rd_v2.py",
            "experiments/exp_generalization/original_mosaic_strict/strict_array_cache.py",
            "code/TOSICA_model_MoE.py",
        ]
    )
    code = sorted(set(code))
    config = _expand_globs(
        root,
        [
            "configs/datasets/**/*",
            "configs/models/**/*",
        ],
    )
    manifests = _expand_globs(
        root,
        [
            "configs/splits/pbmc_cite_seq/nested_lodo/*.csv",
            "configs/splits/pbmc_cite_seq/*manifest*.csv",
            "configs/splits/pbmc_cite_seq/pseudo_unknown_*.csv",
        ],
    )
    testdata = _expand_globs(
        root,
        ["experiments/reproducibility/testdata/**/*"],
    )
    metadata = [
        path
        for path in ["AGENTS.md", "README.md", "LICENSE", "LICENSE.txt"]
        if (root / path).is_file()
    ]
    return {
        "code": code,
        "config": config,
        "manifest": manifests,
        "testdata": testdata,
        "evidence": list(DEFAULT_REQUIRED_ARTIFACTS),
        "metadata": metadata,
    }


def collect_environment_metadata(root: Path) -> Dict[str, object]:
    packages = {
        distribution.metadata["Name"]: distribution.version
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    }
    metadata: Dict[str, object] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "conda_prefix": os.environ.get("CONDA_PREFIX", ""),
        "packages": dict(sorted(packages.items(), key=lambda item: item[0].lower())),
    }
    try:
        metadata["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        metadata["git_dirty"] = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        metadata["git_commit"] = ""
        metadata["git_dirty"] = None
    try:
        metadata["gpu"] = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip().splitlines()
    except (
        OSError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ):
        metadata["gpu"] = []
    return metadata


def _validate_required_artifacts(
    root: Path,
    required_artifacts: Sequence[str],
) -> None:
    missing = []
    for value in required_artifacts:
        relative_path = _normalise_relative_path(root, value)
        source = root / relative_path
        if not source.is_file() or source.stat().st_size == 0:
            missing.append(relative_path.as_posix())
    if missing:
        raise ReleaseBlockedError(
            "missing required evidence: " + ", ".join(sorted(missing))
        )


def _write_checksums(output_dir: Path) -> None:
    rows = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "CHECKSUMS.sha256":
            relative_path = path.relative_to(output_dir).as_posix()
            rows.append(f"{sha256_file(path)}  {relative_path}")
    (output_dir / "CHECKSUMS.sha256").write_text(
        "\n".join(rows) + "\n",
        encoding="utf-8",
    )


def build_release_candidate(
    root: Path = ROOT,
    output_dir: Path = DEFAULT_OUTPUT,
    required_artifacts: Optional[Sequence[str]] = None,
    source_groups: Optional[Mapping[str, Sequence[str]]] = None,
    command_lines: Optional[Sequence[str]] = None,
    environment_metadata: Optional[Mapping[str, object]] = None,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
) -> pd.DataFrame:
    """Build a lightweight release directory after all required evidence exists."""

    root = root.resolve()
    output_dir = output_dir.resolve()
    required = list(
        DEFAULT_REQUIRED_ARTIFACTS
        if required_artifacts is None
        else required_artifacts
    )
    _validate_required_artifacts(root, required)
    if output_dir.exists():
        raise ReleaseBlockedError(f"release output already exists: {output_dir}")
    staging = output_dir.with_name(output_dir.name + ".tmp")
    if staging.exists():
        raise ReleaseBlockedError(f"stale release staging directory exists: {staging}")

    groups = (
        default_source_groups(root)
        if source_groups is None
        else {key: list(value) for key, value in source_groups.items()}
    )
    commands = list(DEFAULT_COMMANDS if command_lines is None else command_lines)
    environment = dict(
        collect_environment_metadata(root)
        if environment_metadata is None
        else environment_metadata
    )
    manifest_rows = []
    exclusion_rows = []
    missing_required_sources = []
    staging.mkdir(parents=True)
    try:
        for category, values in groups.items():
            for value in values:
                relative_path = _normalise_relative_path(root, value)
                source = root / relative_path
                if not source.is_file():
                    exclusion_rows.append(
                        {
                            "category": category,
                            "source_path": relative_path.as_posix(),
                            "reason": "source_missing",
                        }
                    )
                    if category in {"code", "config", "manifest"}:
                        missing_required_sources.append(relative_path.as_posix())
                    continue
                if _is_forbidden(relative_path):
                    exclusion_rows.append(
                        {
                            "category": category,
                            "source_path": relative_path.as_posix(),
                            "reason": "forbidden_path_or_suffix",
                        }
                    )
                    continue
                if source.stat().st_size > max_file_bytes:
                    exclusion_rows.append(
                        {
                            "category": category,
                            "source_path": relative_path.as_posix(),
                            "reason": "exceeds_size_limit",
                        }
                    )
                    continue
                packaged_path = (
                    relative_path
                    if category == "metadata"
                    and relative_path.name in {"LICENSE", "LICENSE.txt"}
                    else Path(category) / relative_path
                )
                destination = staging / packaged_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                manifest_rows.append(
                    {
                        "category": category,
                        "source_path": relative_path.as_posix(),
                        "packaged_path": packaged_path.as_posix(),
                        "size_bytes": destination.stat().st_size,
                        "sha256": sha256_file(destination),
                    }
                )

        packaged_categories = {row["category"] for row in manifest_rows}
        required_categories = {"code", "config", "manifest", "evidence"}
        if source_groups is None:
            required_categories.add("testdata")
        missing_categories = sorted(required_categories - packaged_categories)
        if missing_required_sources or missing_categories:
            details = []
            if missing_required_sources:
                details.append(
                    "missing required source files: "
                    + ", ".join(sorted(missing_required_sources))
                )
            if missing_categories:
                details.append(
                    "empty required release categories: "
                    + ", ".join(missing_categories)
                )
            raise ReleaseBlockedError("; ".join(details))

        manifest = pd.DataFrame(
            manifest_rows,
            columns=[
                "category",
                "source_path",
                "packaged_path",
                "size_bytes",
                "sha256",
            ],
        )
        exclusions = pd.DataFrame(
            exclusion_rows,
            columns=["category", "source_path", "reason"],
        )
        manifest.to_csv(staging / "release_manifest.csv", index=False)
        exclusions.to_csv(staging / "excluded_files.csv", index=False)
        (staging / "commands.txt").write_text(
            "\n".join(commands) + "\n",
            encoding="utf-8",
        )
        (staging / "environment_metadata.json").write_text(
            json.dumps(environment, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (staging / "README.md").write_text(
            "# MOSAIC-N V33 release candidate\n\n"
            "This lightweight package contains executable code, configurations, "
            "split manifests, synthetic smoke-test data, compact evidence tables, "
            "reproduction commands and environment metadata. Run "
            "`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python "
            "code/experiments/reproducibility/v33_release_smoke_test.py "
            "--release-dir .` from this directory to validate the packaged test "
            "fixture. The fixture is synthetic and is not scientific evidence. "
            "Raw participant data, caches, predictions and model checkpoints are "
            "intentionally excluded.\n",
            encoding="utf-8",
        )
        _write_checksums(staging)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        staging.rename(output_dir)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-file-mb", type=float, default=10.0)
    args = parser.parse_args()
    manifest = build_release_candidate(
        root=args.root,
        output_dir=args.output_dir,
        max_file_bytes=int(args.max_file_mb * 1024 * 1024),
    )
    print(
        f"V33 release candidate: {len(manifest)} files at "
        f"{args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
