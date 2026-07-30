#!/usr/bin/env python
"""Build the locked five-target V33 leave-class-out caches."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


ROOT = Path(__file__).resolve().parents[3]
STRICT_DIR = ROOT / "experiments/exp_generalization/original_mosaic_strict"
if str(STRICT_DIR) not in sys.path:
    sys.path.insert(0, str(STRICT_DIR))

from build_pseudo_unknown_cache import (  # noqa: E402
    build_pseudo_unknown_arrays,
    save_arrays,
)
from strict_array_cache import save_strict_arrays_cache  # noqa: E402


DATE = "2026-07-23"
TARGETS = [
    "gdT_2",
    "NK_3",
    "CD4 TCM_1",
    "CD8 TEM_4",
    "B naive lambda",
]


def safe_target_name(target: str) -> str:
    return target.replace(" ", "_").replace("/", "_")


def validation_threshold(
    validation_known_scores: np.ndarray,
    known_coverage: float,
) -> float:
    if not 0.0 < known_coverage <= 1.0:
        raise ValueError("known coverage must be in (0, 1]")
    values = np.asarray(validation_known_scores, dtype=float)
    if len(values) == 0:
        raise ValueError("validation known scores must not be empty")
    return float(np.quantile(values, known_coverage, method="lower"))


def build_known_training_arrays(source: dict) -> dict:
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.asarray(source["label_classes"], dtype=str)
    return {
        "gene": np.asarray(source["gene"], dtype=np.float32),
        "protein": np.asarray(source["protein"], dtype=np.float32),
        "labels": np.asarray(source["labels"], dtype=np.int64),
        "train_idx": np.asarray(source["train_idx"], dtype=np.int64),
        "val_idx": np.asarray(source["val_idx"], dtype=np.int64),
        "test_idx": np.asarray(source["test_known_idx"], dtype=np.int64),
        "label_encoder": label_encoder,
        "gene_names": np.asarray(source["gene_names"], dtype=str).tolist(),
        "protein_names": np.asarray(source["protein_names"], dtype=str).tolist(),
        "cell_ids": np.asarray(source["cell_ids"], dtype=str),
    }


def target_paths(target: str, n_genes: int, max_cells: int) -> dict[str, Path]:
    safe = safe_target_name(target)
    stem = f"{safe}_g{n_genes}_c{max_cells}_seed42"
    base = ROOT / "cache/mosaic_n_v33/unknown"
    return {
        "unknown": base / f"{stem}_unknown_arrays.npz",
        "known": base / f"{stem}_known_training_arrays.npz",
        "metadata": base / f"{stem}_metadata.json",
    }


def build_target(
    target: str,
    n_genes: int,
    max_cells: int,
    force: bool,
) -> dict:
    paths = target_paths(target, n_genes, max_cells)
    if all(path.exists() for path in paths.values()) and not force:
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        return {**metadata, "reused": True, "runtime_seconds": 0.0}
    start = time.perf_counter()
    source, metadata = build_pseudo_unknown_arrays(
        root=ROOT,
        target_label=target,
        n_genes=n_genes,
        max_cells=max_cells,
        seed=42,
        n_quantiles=1000,
    )
    known_arrays = build_known_training_arrays(source)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    save_arrays(source, paths["unknown"])
    protocol_metadata = {
        **metadata,
        "date": DATE,
        "unknown_cache": str(paths["unknown"].relative_to(ROOT)),
        "known_training_cache": str(paths["known"].relative_to(ROOT)),
        "n_train": int(len(source["train_idx"])),
        "n_val": int(len(source["val_idx"])),
        "n_test_known": int(len(source["test_known_idx"])),
        "n_test_unknown": int(len(source["test_unknown_idx"])),
        "n_classes": int(len(source["label_classes"])),
        "threshold_source": "known validation only",
        "test_label_tuning": False,
    }
    save_strict_arrays_cache(
        known_arrays,
        paths["known"],
        metadata=protocol_metadata,
    )
    paths["metadata"].write_text(
        json.dumps(protocol_metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {
        **protocol_metadata,
        "reused": False,
        "runtime_seconds": float(time.perf_counter() - start),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true")
    group.add_argument("--target", choices=TARGETS)
    parser.add_argument("--n-genes", type=int, default=3000)
    parser.add_argument("--max-cells", type=int, default=100000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    targets = TARGETS if args.all else [args.target]
    rows = []
    for target in targets:
        print(f"Building V33 leave-class-out cache: {target}", flush=True)
        rows.append(
            build_target(
                target,
                n_genes=args.n_genes,
                max_cells=args.max_cells,
                force=args.force,
            )
        )
    summary = pd.DataFrame(rows)
    for base in (ROOT / "results/tables", ROOT / "output/tables"):
        base.mkdir(parents=True, exist_ok=True)
        summary.to_csv(
            base / f"mosaic_n_v33_unknown_cache_summary_{DATE}.csv",
            index=False,
        )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
