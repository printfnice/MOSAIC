#!/usr/bin/env python
"""Read-only resource and artifact preflight for the V33 evidence closure."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
DATE = "2026-07-23"
MIN_GPU_MEMORY_MB = 12000
MIN_DISK_GB = 20.0


def _row(check: str, status: str, value: str, criterion: str, action: str) -> dict[str, str]:
    return {
        "check": check,
        "status": status,
        "value": value,
        "criterion": criterion,
        "action": action,
    }


def evaluate_preflight(
    required_paths: dict[str, Path],
    gpu_name: str,
    gpu_memory_free_mb: float,
    disk_free_gb: float,
    conda_envs: set[str],
) -> pd.DataFrame:
    rows = []
    for name, path in required_paths.items():
        exists = Path(path).exists()
        rows.append(
            _row(
                name,
                "pass" if exists else "blocker",
                str(path),
                "required artifact exists",
                "" if exists else "restore or rebuild the required artifact before training",
            )
        )
    gpu_ok = bool(gpu_name.strip()) and gpu_memory_free_mb >= MIN_GPU_MEMORY_MB
    rows.append(
        _row(
            "gpu_memory",
            "pass" if gpu_ok else "blocker",
            f"{gpu_name or 'unavailable'}; free={gpu_memory_free_mb:.0f} MiB",
            f"CUDA GPU with at least {MIN_GPU_MEMORY_MB} MiB free",
            "" if gpu_ok else "free GPU memory or move the run to a compatible device",
        )
    )
    disk_ok = disk_free_gb >= MIN_DISK_GB
    rows.append(
        _row(
            "data_disk",
            "pass" if disk_ok else "blocker",
            f"free={disk_free_gb:.2f} GiB",
            f"at least {MIN_DISK_GB:.1f} GiB free",
            "" if disk_ok else "free data-disk space before generating fold caches",
        )
    )
    for env_name in ("TOSICA", "MMoCHi"):
        present = env_name in conda_envs
        rows.append(
            _row(
                f"env_{env_name}",
                "pass" if present else "blocker",
                env_name,
                "named conda environment is installed",
                "" if present else f"restore the {env_name} environment",
            )
        )
    return pd.DataFrame(rows)


def _gpu_state() -> tuple[str, float]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        first = result.stdout.strip().splitlines()[0]
        name, memory = [part.strip() for part in first.rsplit(",", 1)]
        return name, float(memory)
    except (FileNotFoundError, subprocess.CalledProcessError, IndexError, ValueError):
        return "", 0.0


def _conda_env_names() -> set[str]:
    conda_executable = os.environ.get("CONDA_EXE", "conda")
    try:
        result = subprocess.run(
            [conda_executable, "env", "list", "--json"],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(result.stdout)
    except (FileNotFoundError, subprocess.CalledProcessError, json.JSONDecodeError):
        return set()
    return {Path(path).name for path in payload.get("envs", [])}


def collect_preflight(root: Path = ROOT) -> pd.DataFrame:
    required_paths = {
        "pbmc_gene": root / "data/pbmc/pbmc_gene.h5ad",
        "pbmc_protein": root / "data/pbmc/pbmc_protein.h5ad",
        "pbmc_lodo_manifest": root / "configs/splits/pbmc_cite_seq/leave_one_donor_out_manifest.csv",
        "pbmc_hierarchy_map": root / "configs/datasets/pbmc_cite_seq/strict_l3_to_l2_l1_map_seed42.csv",
        "pdc101": root / "data/gse229791_mmochi/pdc101_sorted_tnk.h5ad",
        "mmochi_result": (
            root
            / "results/exp_generalization/mmochi_pdc101_sorted_ext_holdout_thresholds/results_summary.csv"
        ),
    }
    gpu_name, gpu_memory = _gpu_state()
    disk_free = shutil.disk_usage(root / "results").free / (1024**3)
    return evaluate_preflight(
        required_paths=required_paths,
        gpu_name=gpu_name,
        gpu_memory_free_mb=gpu_memory,
        disk_free_gb=disk_free,
        conda_envs=_conda_env_names(),
    )


def write_outputs(frame: pd.DataFrame, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"mosaic_n_v33_preflight_{DATE}.csv"
    json_path = out_dir / f"mosaic_n_v33_preflight_{DATE}.json"
    frame.to_csv(csv_path, index=False)
    summary = {
        "date": DATE,
        "pass_count": int(frame["status"].eq("pass").sum()),
        "blocker_count": int(frame["status"].eq("blocker").sum()),
        "checks": frame.to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return csv_path, json_path


def main() -> None:
    frame = collect_preflight(ROOT)
    write_outputs(frame, ROOT / "results/tables")
    write_outputs(frame, ROOT / "output/tables")
    print(frame.to_string(index=False))
    if frame["status"].eq("blocker").any():
        raise SystemExit(2)


if __name__ == "__main__":
    main()
