#!/usr/bin/env python
"""Train the locked V33 five-target leave-class-out model matrix."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[3]
LOCAL_DIR = Path(__file__).resolve().parent
GENERALIZATION_V33 = ROOT / "experiments/exp_generalization/mosaic_n_v33"
for directory in (LOCAL_DIR, GENERALIZATION_V33):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from build_v33_unknown_protocol import (  # noqa: E402
    TARGETS,
    safe_target_name,
    target_paths,
)
from run_v33_donor_matrix import (  # noqa: E402
    _run_command,
    build_mlp_command,
    build_mosaic_command,
    materialize_mlp_support_artifacts,
    run_is_complete,
    teacher_validation_references,
    write_artifact_index,
)


DATE = "2026-07-23"
SEEDS = [41, 42, 43]


def expected_unknown_run_keys(
    targets: list[str],
    seeds: list[int],
) -> list[tuple[str, int, str]]:
    rows = []
    for target in targets:
        rows.append((target, 42, "mlp_teacher"))
        rows.extend((target, seed, "mosaic_full") for seed in seeds)
    return rows


def execute_unknown_matrix(
    base_dir: Path,
    targets: list[str],
    seeds: list[int],
    epochs: int,
    n_genes: int,
    max_cells: int,
) -> pd.DataFrame:
    base_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "date": DATE,
        "protocol": "five-target leave-class-out stress test",
        "targets": targets,
        "seeds": seeds,
        "epochs": epochs,
        "known_coverage_targets": [0.95, 0.80],
        "threshold_source": "known validation only",
        "test_label_tuning": False,
        "claim_boundary": "pseudo-unknown stress test, not prospective open-set proof",
    }
    (base_dir / "protocol.yaml").write_text(
        yaml.safe_dump(protocol, sort_keys=False),
        encoding="utf-8",
    )
    rows = []
    for target in targets:
        paths = target_paths(target, n_genes, max_cells)
        cache_path = paths["known"]
        if not cache_path.exists():
            raise FileNotFoundError(cache_path)
        target_dir = base_dir / safe_target_name(target)
        teacher_dir = target_dir / "mlp_seed42"
        teacher_probabilities = teacher_dir / "probabilities_train.csv"
        if not run_is_complete(teacher_dir, "mlp") or not teacher_probabilities.exists():
            runtime = _run_command(
                build_mlp_command(
                    python=Path(sys.executable),
                    cache_path=cache_path,
                    out_dir=teacher_dir,
                    seed=42,
                    epochs=epochs,
                    save_teacher=True,
                ),
                teacher_dir,
            )
            materialize_mlp_support_artifacts(teacher_dir)
            write_artifact_index(teacher_dir)
            status = "completed"
        else:
            runtime, status = 0.0, "reused"
        rows.append(
            {
                "target_label": target,
                "seed": 42,
                "method": "mlp_teacher",
                "status": status,
                "runtime_seconds": runtime,
                "out_dir": str(teacher_dir.relative_to(ROOT)),
            }
        )
        reference_accuracy, reference_weighted_f1 = teacher_validation_references(
            teacher_dir
        )
        for seed in seeds:
            out_dir = target_dir / f"mosaic_full_seed{seed}"
            if run_is_complete(out_dir, "mosaic_full"):
                runtime, status = 0.0, "reused"
            else:
                runtime = _run_command(
                    build_mosaic_command(
                        python=Path(sys.executable),
                        cache_path=cache_path,
                        teacher_path=teacher_probabilities,
                        out_dir=out_dir,
                        seed=seed,
                        epochs=epochs,
                        method="mosaic_full",
                        selection_reference_accuracy=reference_accuracy,
                        selection_reference_weighted_f1=reference_weighted_f1,
                    ),
                    out_dir,
                )
                write_artifact_index(out_dir)
                status = "completed"
            rows.append(
                {
                    "target_label": target,
                    "seed": seed,
                    "method": "mosaic_full",
                    "status": status,
                    "runtime_seconds": runtime,
                    "out_dir": str(out_dir.relative_to(ROOT)),
                }
            )
        pd.DataFrame(rows).to_csv(base_dir / "run_status.csv", index=False)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true")
    group.add_argument("--target", choices=TARGETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--n-genes", type=int, default=3000)
    parser.add_argument("--max-cells", type=int, default=100000)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    targets = TARGETS if args.all else [args.target]
    base_name = "mosaic_n_v33_smoke" if args.smoke else "mosaic_n_v33"
    base_dir = ROOT / f"results/exp_unknown_celltype/{base_name}"
    status = execute_unknown_matrix(
        base_dir=base_dir,
        targets=targets,
        seeds=args.seeds,
        epochs=2 if args.smoke else 35,
        n_genes=args.n_genes,
        max_cells=args.max_cells,
    )
    status.to_csv(base_dir / "run_status.csv", index=False)
    print(status.to_string(index=False))


if __name__ == "__main__":
    main()
