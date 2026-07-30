#!/usr/bin/env python
"""Orchestrate the locked V33 PBMC nested-donor training matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, classification_report, f1_score


ROOT = Path(__file__).resolve().parents[3]
DATE = "2026-07-23"
DONORS = [f"P{index}" for index in range(1, 9)]
DEFAULT_SEEDS = [41, 42, 43]
FORMAL_METHODS = ["mlp", "mosaic_full", "mosaic_no_hsr", "mosaic_no_kd"]
MLP_SCRIPT = ROOT / "experiments/exp_generalization/original_mosaic_strict/run_mlp_strict_baseline.py"
MOSAIC_SCRIPT = ROOT / "experiments/exp_generalization/mosaic_n_v16/run_v16_hsr_distill.py"


def expected_run_keys(
    donors: list[str],
    seeds: list[int],
    methods: list[str],
) -> list[tuple[str, int, str]]:
    return [
        (donor, seed, method)
        for donor in donors
        for seed in seeds
        for method in methods
    ]


def build_mlp_command(
    python: Path,
    cache_path: Path,
    out_dir: Path,
    seed: int,
    epochs: int,
    save_teacher: bool,
) -> list[str]:
    command = [
        str(python),
        str(MLP_SCRIPT),
        "--out-dir",
        str(out_dir),
        "--seed",
        str(seed),
        "--cache-path",
        str(cache_path),
        "--modality",
        "both",
        "--hidden-dims",
        "512,128",
        "--epochs",
        str(epochs),
        "--batch-size",
        "1024",
        "--lr",
        "0.001",
        "--weight-decay",
        "0.0001",
        "--dropout",
        "0.2",
        "--label-smoothing",
        "0.03",
        "--patience",
        "8",
        "--grad-clip",
        "5.0",
    ]
    if save_teacher:
        command.append("--save-train-probabilities")
    return command


def build_mosaic_command(
    python: Path,
    cache_path: Path,
    teacher_path: Path,
    out_dir: Path,
    seed: int,
    epochs: int,
    method: str,
    selection_reference_accuracy: float,
    selection_reference_weighted_f1: float,
    hsr_sibling_groups: list[list[str]] | None = None,
) -> list[str]:
    if method not in {"mosaic_full", "mosaic_no_hsr", "mosaic_no_kd"}:
        raise ValueError(f"unsupported MOSAIC-N method: {method}")
    hsr_enabled = method != "mosaic_no_hsr"
    distill_alpha = 0.0 if method == "mosaic_no_kd" else 0.05
    command = [
        str(python),
        str(MOSAIC_SCRIPT),
        "--out-dir",
        str(out_dir),
        "--teacher-train-probabilities",
        str(teacher_path),
        "--cache-path",
        str(cache_path),
        "--seed",
        str(seed),
        "--n-genes",
        "3000",
        "--max-cells",
        "0",
        "--hidden-dim",
        "256",
        "--encoder-hidden-dims",
        "512",
        "--fusion-hidden-dims",
        "512,256",
        "--epochs",
        str(epochs),
        "--batch-size",
        "1024",
        "--lr",
        "0.001",
        "--weight-decay",
        "0.0001",
        "--dropout",
        "0.2",
        "--label-smoothing",
        "0.03",
        "--modality-dropout",
        "0.15",
        "--branch-loss-weight",
        "0.35",
        "--fusion-loss-weight",
        "0.5",
        "--gate-temperature",
        "1.0",
        "--head-type",
        "linear",
        "--patience",
        "14",
        "--grad-clip",
        "5.0",
        "--distill-alpha",
        str(distill_alpha),
        "--distill-temperature",
        "2.0",
        "--ce-class-weight-mode",
        "sqrt_balanced",
        "--ce-class-weight-max",
        "5.0",
        "--kd-class-weight-mode",
        "sqrt_balanced",
        "--kd-class-weight-max",
        "5.0",
        "--selection-metric",
        "constraint_macro",
        "--selection-acc-tolerance",
        "0.0015",
        "--selection-weighted-tolerance",
        "0.0015",
        "--selection-constraint-penalty",
        "2.0",
        "--selection-reference-accuracy",
        str(selection_reference_accuracy),
        "--selection-reference-weighted-f1",
        str(selection_reference_weighted_f1),
        "--hsr-mode",
        "hierarchy" if hsr_enabled else "off",
        "--hsr-loss-weight",
        "0.2" if hsr_enabled else "0.0",
    ]
    if hsr_enabled:
        command.extend(
            [
                "--hsr-gate-floor",
                "0.02",
                "--hsr-gate-max",
                "0.2",
            ]
        )
        for group in hsr_sibling_groups or []:
            command.extend(["--hsr-sibling-group", "|".join(group)])
    return command


def run_is_complete(out_dir: Path, method: str) -> bool:
    common = {
        "config.json",
        "model.pt",
        "results_summary.csv",
        "per_class_metrics.csv",
        "run.log",
    }
    if method == "mlp":
        required = common | {"predictions.csv", "probabilities_val.csv", "probabilities_test.csv"}
    else:
        required = common | {"predictions_full.csv", "probabilities_test.csv"}
    return all((out_dir / name).exists() for name in required)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_artifact_index(out_dir: Path) -> None:
    rows = []
    for path in sorted(out_dir.iterdir()):
        if not path.is_file() or path.name == "artifact_index.csv":
            continue
        rows.append(
            {
                "artifact": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "artifact_index.csv", index=False)


def materialize_mlp_support_artifacts(out_dir: Path) -> None:
    predictions = pd.read_csv(out_dir / "predictions.csv")
    report = classification_report(
        predictions["label"].astype(str),
        predictions["prediction"].astype(str),
        labels=sorted(predictions["label"].astype(str).unique()),
        output_dict=True,
        zero_division=0,
    )
    rows = []
    for label in sorted(predictions["label"].astype(str).unique()):
        metrics = report[label]
        rows.append(
            {
                "class_label": label,
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1": float(metrics["f1-score"]),
                "support": int(metrics["support"]),
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "per_class_metrics.csv", index=False)
    subprocess_log = out_dir / "subprocess.log"
    if subprocess_log.exists():
        shutil.copyfile(subprocess_log, out_dir / "run.log")


def teacher_validation_references(teacher_dir: Path) -> tuple[float, float]:
    frame = pd.read_csv(teacher_dir / "probabilities_val.csv")
    accuracy = float(accuracy_score(frame["label"], frame["prediction"]))
    weighted_f1 = float(
        f1_score(
            frame["label"],
            frame["prediction"],
            average="weighted",
            zero_division=0,
        )
    )
    return accuracy, weighted_f1


def _run_command(command: list[str], out_dir: Path) -> float:
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "4"
    environment["MKL_NUM_THREADS"] = "4"
    with (out_dir / "subprocess.log").open("a", encoding="utf-8") as log_handle:
        log_handle.write("$ " + " ".join(command) + "\n")
        log_handle.flush()
        subprocess.run(
            command,
            cwd=ROOT,
            check=True,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=environment,
        )
    return float(time.perf_counter() - start)


def _fold_cache(donor: str) -> Path:
    validation = f"P{int(donor[1:]) % 8 + 1}"
    return ROOT / f"cache/mosaic_n_v33/test_{donor}_val_{validation}_g3000.npz"


def _run_dir(base: Path, donor: str, method: str, seed: int) -> Path:
    return base / f"test_{donor}" / f"{method}_seed{seed}"


def write_protocol_files(base: Path, donors: list[str], seeds: list[int], epochs: int) -> None:
    base.mkdir(parents=True, exist_ok=True)
    protocol = {
        "date": DATE,
        "protocol": "PBMC nested leave-one-donor-out",
        "test_donors": donors,
        "validation_mapping": {donor: f"P{int(donor[1:]) % 8 + 1}" for donor in donors},
        "seeds": seeds,
        "methods": FORMAL_METHODS,
        "epochs": epochs,
        "teacher": "per-fold seed-42 MLP training probabilities",
        "test_label_tuning": False,
        "statistical_unit": "held-out donor",
    }
    (base / "protocol.yaml").write_text(
        yaml.safe_dump(protocol, sort_keys=False),
        encoding="utf-8",
    )
    (base / "README.md").write_text(
        "# V33 PBMC nested donor matrix\n\n"
        "Every fold uses a locked test donor, the next donor as validation, and "
        "the remaining six donors for train-only feature selection and scaling. "
        "Model selection never uses test labels.\n",
        encoding="utf-8",
    )


def execute_matrix(
    base: Path,
    donors: list[str],
    seeds: list[int],
    methods: list[str],
    epochs: int,
) -> pd.DataFrame:
    write_protocol_files(base, donors, seeds, epochs)
    rows = []
    for donor in donors:
        cache_path = _fold_cache(donor)
        if not cache_path.exists():
            raise FileNotFoundError(cache_path)
        teacher_dir = _run_dir(base, donor, "mlp", 42)
        teacher_path = teacher_dir / "probabilities_train.csv"
        if not run_is_complete(teacher_dir, "mlp") or not teacher_path.exists():
            command = build_mlp_command(
                Path(sys.executable),
                cache_path,
                teacher_dir,
                seed=42,
                epochs=epochs,
                save_teacher=True,
            )
            runtime = _run_command(command, teacher_dir)
            materialize_mlp_support_artifacts(teacher_dir)
            write_artifact_index(teacher_dir)
            rows.append(
                {
                    "test_donor": donor,
                    "seed": 42,
                    "method": "mlp",
                    "status": "completed",
                    "runtime_seconds": runtime,
                    "out_dir": str(teacher_dir.relative_to(ROOT)),
                }
            )
        else:
            rows.append(
                {
                    "test_donor": donor,
                    "seed": 42,
                    "method": "mlp",
                    "status": "reused",
                    "runtime_seconds": 0.0,
                    "out_dir": str(teacher_dir.relative_to(ROOT)),
                }
            )
        reference_accuracy, reference_weighted_f1 = teacher_validation_references(
            teacher_dir
        )

        if "mlp" in methods:
            for seed in seeds:
                if seed == 42:
                    continue
                out_dir = _run_dir(base, donor, "mlp", seed)
                if run_is_complete(out_dir, "mlp"):
                    status, runtime = "reused", 0.0
                else:
                    command = build_mlp_command(
                        Path(sys.executable),
                        cache_path,
                        out_dir,
                        seed=seed,
                        epochs=epochs,
                        save_teacher=False,
                    )
                    runtime = _run_command(command, out_dir)
                    materialize_mlp_support_artifacts(out_dir)
                    write_artifact_index(out_dir)
                    status = "completed"
                rows.append(
                    {
                        "test_donor": donor,
                        "seed": seed,
                        "method": "mlp",
                        "status": status,
                        "runtime_seconds": runtime,
                        "out_dir": str(out_dir.relative_to(ROOT)),
                    }
                )

        for method in methods:
            if method == "mlp":
                continue
            for seed in seeds:
                out_dir = _run_dir(base, donor, method, seed)
                if run_is_complete(out_dir, method):
                    status, runtime = "reused", 0.0
                else:
                    command = build_mosaic_command(
                        python=Path(sys.executable),
                        cache_path=cache_path,
                        teacher_path=teacher_path,
                        out_dir=out_dir,
                        seed=seed,
                        epochs=epochs,
                        method=method,
                        selection_reference_accuracy=reference_accuracy,
                        selection_reference_weighted_f1=reference_weighted_f1,
                    )
                    runtime = _run_command(command, out_dir)
                    write_artifact_index(out_dir)
                    status = "completed"
                rows.append(
                    {
                        "test_donor": donor,
                        "seed": seed,
                        "method": method,
                        "status": status,
                        "runtime_seconds": runtime,
                        "out_dir": str(out_dir.relative_to(ROOT)),
                    }
                )
        status_frame = pd.DataFrame(rows)
        status_frame.to_csv(base / "run_status.csv", index=False)
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true")
    group.add_argument("--test-donor", choices=DONORS)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--methods", nargs="+", choices=FORMAL_METHODS, default=FORMAL_METHODS)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    donors = DONORS if args.all else [args.test_donor]
    epochs = 2 if args.smoke else 35
    base_name = "donor_matrix_smoke" if args.smoke else "donor_matrix"
    base = ROOT / f"results/exp_generalization/mosaic_n_v33/{base_name}"
    status = execute_matrix(
        base=base,
        donors=donors,
        seeds=args.seeds,
        methods=args.methods,
        epochs=epochs,
    )
    status.to_csv(base / "run_status.csv", index=False)
    print(status.to_string(index=False))


if __name__ == "__main__":
    main()
