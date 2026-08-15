#!/usr/bin/env python
"""Run MOSAIC-N on the locked official-MMoCHi PDC101 external holdout."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[3]
STRICT_DIR = ROOT / "experiments/exp_generalization/original_mosaic_strict"
V33_DIR = Path(__file__).resolve().parent
for directory in (STRICT_DIR, V33_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from run_v33_donor_matrix import (  # noqa: E402
    _run_command,
    build_mlp_command,
    build_mosaic_command,
    materialize_mlp_support_artifacts,
    run_is_complete,
    teacher_validation_references,
    write_artifact_index,
)
from strict_array_cache import save_strict_arrays_cache  # noqa: E402


DATE = "2026-07-23"
SEEDS = [41, 42, 43]
PDC_HSR_GROUPS = [
    ["cd4_n", "cd4_cm", "cd4_em"],
    ["cd8_n", "cd8_cm", "cd8_em", "cd8_emra"],
]


def validate_locked_holdout(
    cell_ids: np.ndarray,
    labels: np.ndarray,
    holdout: pd.DataFrame,
) -> np.ndarray:
    required = {"MMoCHi_obs_names", "sort_label", "external_holdout"}
    missing = required.difference(holdout.columns)
    if missing:
        raise ValueError(f"PDC101 holdout missing columns: {sorted(missing)}")
    holdout_ids = holdout["MMoCHi_obs_names"].astype(str)
    if holdout_ids.duplicated().any():
        raise ValueError("PDC101 holdout contains duplicate cell IDs")
    external = holdout["external_holdout"]
    if not external.astype(str).str.lower().eq("true").all():
        raise ValueError("PDC101 artifact is not entirely external_holdout=True")
    label_by_cell = pd.Series(
        np.asarray(labels, dtype=str),
        index=np.asarray(cell_ids, dtype=str),
    )
    missing_ids = sorted(set(holdout_ids) - set(label_by_cell.index))
    if missing_ids:
        raise ValueError(
            f"PDC101 holdout IDs are absent from h5ad: {missing_ids[:3]}"
        )
    source_labels = label_by_cell.loc[holdout_ids].to_numpy(dtype=str)
    artifact_labels = holdout["sort_label"].astype(str).to_numpy()
    if not np.array_equal(source_labels, artifact_labels):
        raise ValueError("PDC101 holdout truth labels do not match the h5ad")
    holdout_id_set = set(holdout_ids)
    return np.asarray(
        [str(cell_id) in holdout_id_set for cell_id in cell_ids],
        dtype=bool,
    )


def allowed_protein_mask(names: list[str]) -> np.ndarray:
    allowed = []
    for name in names:
        normalized = str(name).lower().replace("_", "-")
        hto = normalized.startswith("hto")
        explicit_control = any(
            token in normalized for token in ("isotype", "ctrl", "control")
        )
        species_igg_control = bool(
            re.search(r"(mouse|rat|armenian-hamster).*igg", normalized)
        )
        allowed.append(not (hto or explicit_control or species_igg_control))
    return np.asarray(allowed, dtype=bool)


def split_train_validation(
    labels: np.ndarray,
    holdout_mask: np.ndarray,
    seed: int,
    validation_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=str)
    holdout_mask = np.asarray(holdout_mask, dtype=bool)
    test_idx = np.flatnonzero(holdout_mask)
    candidate = np.flatnonzero(~holdout_mask)
    train_idx, val_idx = train_test_split(
        candidate,
        test_size=validation_fraction,
        stratify=labels[candidate],
        random_state=seed,
    )
    return (
        np.sort(train_idx.astype(np.int64)),
        np.sort(val_idx.astype(np.int64)),
        np.sort(test_idx.astype(np.int64)),
    )


def _dense_float32(matrix) -> np.ndarray:
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def build_pdc_cache(
    dataset_path: Path,
    holdout_path: Path,
    cache_path: Path,
    n_genes: int,
    force: bool,
) -> dict:
    metadata_path = cache_path.with_name(cache_path.stem + "_metadata.json")
    if cache_path.exists() and metadata_path.exists() and not force:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    start = time.perf_counter()
    adata = ad.read_h5ad(dataset_path)
    holdout = pd.read_csv(holdout_path)
    labels_text = adata.obs["sort_label"].astype(str).to_numpy()
    holdout_mask = validate_locked_holdout(
        adata.obs_names.astype(str).to_numpy(),
        labels_text,
        holdout,
    )
    if int(holdout_mask.sum()) != len(holdout):
        raise ValueError("PDC101 holdout IDs do not match the locked MMoCHi artifact")
    train_idx, val_idx, test_idx = split_train_validation(
        labels_text,
        holdout_mask,
        seed=42,
        validation_fraction=0.15,
    )
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels_text).astype(np.int64)

    train_gene = sparse.csr_matrix(adata.X[train_idx], dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        scores, _ = f_classif(train_gene, labels[train_idx])
    scores = np.nan_to_num(scores, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
    selected_gene_idx = np.sort(np.argsort(scores)[-min(n_genes, adata.n_vars) :])
    gene_matrix = _dense_float32(adata.X[:, selected_gene_idx])

    protein_frame = adata.obsm["protein"]
    protein_names_all = [str(value) for value in protein_frame.columns]
    protein_keep = allowed_protein_mask(protein_names_all)
    protein_names = np.asarray(protein_names_all, dtype=str)[protein_keep]
    protein_matrix = _dense_float32(
        protein_frame.loc[:, protein_keep].to_numpy(dtype=np.float32)
    )
    if any(
        name.upper().startswith("HTO")
        or "ISOTYPE" in name.upper()
        or "CTRL" in name.upper()
        for name in protein_names
    ):
        raise ValueError("HTO or control protein leaked into PDC101 features")

    gene_scaler = StandardScaler()
    protein_scaler = StandardScaler()
    gene_scaler.fit(gene_matrix[train_idx])
    protein_scaler.fit(protein_matrix[train_idx])
    for indices in (train_idx, val_idx, test_idx):
        gene_matrix[indices] = gene_scaler.transform(gene_matrix[indices])
        protein_matrix[indices] = protein_scaler.transform(protein_matrix[indices])

    arrays = {
        "gene": gene_matrix.astype(np.float32),
        "protein": protein_matrix.astype(np.float32),
        "labels": labels,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "label_encoder": label_encoder,
        "gene_names": adata.var_names[selected_gene_idx].astype(str).tolist(),
        "protein_names": protein_names.tolist(),
        "cell_ids": adata.obs_names.astype(str).to_numpy(),
    }
    metadata = {
        "date": DATE,
        "dataset": str(dataset_path.relative_to(ROOT)),
        "holdout_artifact": str(holdout_path.relative_to(ROOT)),
        "n_cells": int(adata.n_obs),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "n_genes": int(gene_matrix.shape[1]),
        "n_proteins": int(protein_matrix.shape[1]),
        "n_classes": int(len(label_encoder.classes_)),
        "excluded_proteins": [
            name for name, keep in zip(protein_names_all, protein_keep) if not keep
        ],
        "feature_selection_fit": "training cells only",
        "scalers_fit": "training cells only",
        "input_scale": (
            "processed h5ad RNA (uns.log1p present) and processed obsm.protein; "
            "no repeated log1p or quantile transform"
        ),
        "holdout_used_for_selection": False,
        "runtime_seconds": float(time.perf_counter() - start),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    save_strict_arrays_cache(arrays, cache_path, metadata=metadata)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    split_path = ROOT / "configs/splits/gse229791_mmochi/pdc101_v33_locked_split.csv"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "cell_id": arrays["cell_ids"],
            "label": labels_text,
            "split": np.where(
                holdout_mask,
                "test",
                np.where(
                    np.isin(np.arange(len(labels)), val_idx),
                    "val",
                    "train",
                ),
            ),
        }
    ).to_csv(split_path, index=False)
    return metadata


def execute_training(
    base_dir: Path,
    cache_path: Path,
    seeds: list[int],
    epochs: int,
) -> pd.DataFrame:
    base_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    teacher_dir = base_dir / "mlp_seed42"
    teacher_path = teacher_dir / "probabilities_train.csv"
    if not run_is_complete(teacher_dir, "mlp") or not teacher_path.exists():
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
        out_dir = base_dir / f"mosaic_full_seed{seed}"
        if run_is_complete(out_dir, "mosaic_full"):
            runtime, status = 0.0, "reused"
        else:
            runtime = _run_command(
                build_mosaic_command(
                    python=Path(sys.executable),
                    cache_path=cache_path,
                    teacher_path=teacher_path,
                    out_dir=out_dir,
                    seed=seed,
                    epochs=epochs,
                    method="mosaic_full",
                    selection_reference_accuracy=reference_accuracy,
                    selection_reference_weighted_f1=reference_weighted_f1,
                    hsr_sibling_groups=PDC_HSR_GROUPS,
                ),
                out_dir,
            )
            write_artifact_index(out_dir)
            status = "completed"
        rows.append(
            {
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
    parser.add_argument("--n-genes", type=int, default=1000)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force-cache", action="store_true")
    args = parser.parse_args()
    dataset = ROOT / "data/gse229791_mmochi/pdc101_sorted_tnk.h5ad"
    holdout = (
        ROOT
        / "results/exp_generalization/mmochi_pdc101_sorted_ext_holdout_thresholds/holdout_predictions.csv"
    )
    cache = ROOT / f"cache/mosaic_n_v33/pdc101_locked_g{args.n_genes}.npz"
    metadata = build_pdc_cache(
        dataset,
        holdout,
        cache,
        n_genes=args.n_genes,
        force=args.force_cache,
    )
    base_name = "pdc101_smoke" if args.smoke else "pdc101"
    base_dir = ROOT / f"results/exp_generalization/mosaic_n_v33/{base_name}"
    status = execute_training(
        base_dir,
        cache,
        args.seeds,
        epochs=2 if args.smoke else 50,
    )
    (base_dir / "config.json").write_text(
        json.dumps(
            {
                "date": DATE,
                "cache_metadata": metadata,
                "seeds": args.seeds,
                "hsr_sibling_groups": PDC_HSR_GROUPS,
                "input_exclusions": "HTO and isotype/control proteins",
                "test_label_tuning": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    status.to_csv(base_dir / "run_status.csv", index=False)
    print(status.to_string(index=False))


if __name__ == "__main__":
    main()
