#!/usr/bin/env python
"""Build leakage-safe PBMC nested leave-one-donor-out array caches."""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import anndata as ad
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import LabelEncoder, QuantileTransformer, StandardScaler


ROOT = Path(__file__).resolve().parents[3]
STRICT_DIR = ROOT / "experiments/exp_generalization/original_mosaic_strict"
if str(STRICT_DIR) not in sys.path:
    sys.path.insert(0, str(STRICT_DIR))

from strict_array_cache import load_strict_arrays_cache, save_strict_arrays_cache  # noqa: E402


DONOR_ORDER = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"]
DATE = "2026-07-23"


def fixed_validation_donor(test_donor: str) -> str:
    if test_donor not in DONOR_ORDER:
        raise ValueError(f"unknown PBMC donor: {test_donor}")
    index = DONOR_ORDER.index(test_donor)
    return DONOR_ORDER[(index + 1) % len(DONOR_ORDER)]


def assign_nested_splits(
    donors: np.ndarray,
    test_donor: str,
    validation_donor: str,
) -> np.ndarray:
    if test_donor == validation_donor:
        raise ValueError("test and validation donors must differ")
    donors = np.asarray(donors, dtype=str)
    observed = set(donors)
    missing = {test_donor, validation_donor}.difference(observed)
    if missing:
        raise ValueError(f"split donors are absent from data: {sorted(missing)}")
    splits = np.full(len(donors), "train", dtype="<U5")
    splits[donors == validation_donor] = "val"
    splits[donors == test_donor] = "test"
    return splits


def build_label_support(labels: np.ndarray, splits: np.ndarray) -> pd.DataFrame:
    labels = np.asarray(labels, dtype=str)
    splits = np.asarray(splits, dtype=str)
    rows = []
    for label in sorted(set(labels)):
        label_mask = labels == label
        counts = {
            split: int(np.sum(label_mask & (splits == split)))
            for split in ("train", "val", "test")
        }
        known = counts["train"] > 0
        rows.append(
            {
                "class_label": label,
                "n_train": counts["train"],
                "n_val": counts["val"],
                "n_test": counts["test"],
                "known_to_train": known,
                "natural_unknown_in_test": (not known) and counts["test"] > 0,
            }
        )
    return pd.DataFrame(rows)


def validate_nested_arrays(
    arrays: dict,
    test_donor: str,
    validation_donor: str,
) -> None:
    n_cells = len(arrays["labels"])
    for key in ("gene", "protein", "donors", "cell_ids"):
        if len(arrays[key]) != n_cells:
            raise ValueError(f"{key} length does not match labels")
    index_sets = {
        name: set(np.asarray(arrays[f"{name}_idx"], dtype=int).tolist())
        for name in ("train", "val", "test")
    }
    if index_sets["train"] & index_sets["val"]:
        raise ValueError("train and validation indices overlap")
    if index_sets["train"] & index_sets["test"]:
        raise ValueError("train and test indices overlap")
    if index_sets["val"] & index_sets["test"]:
        raise ValueError("validation and test indices overlap")
    combined = index_sets["train"] | index_sets["val"] | index_sets["test"]
    if combined != set(range(n_cells)):
        raise ValueError("nested split does not cover every cell exactly once")

    donors = np.asarray(arrays["donors"], dtype=str)
    if np.any(donors[arrays["train_idx"]] == test_donor):
        raise ValueError("test donor leaked into training indices")
    if np.any(donors[arrays["val_idx"]] == test_donor):
        raise ValueError("test donor leaked into validation indices")
    if set(donors[arrays["test_idx"]]) != {test_donor}:
        raise ValueError("test indices contain a donor other than the locked test donor")
    if set(donors[arrays["val_idx"]]) != {validation_donor}:
        raise ValueError("validation indices do not match the locked validation donor")


def _log1p_sparse(matrix) -> sparse.csr_matrix:
    matrix = sparse.csr_matrix(matrix, dtype=np.float32)
    matrix.data = np.log1p(np.clip(matrix.data, 0.0, None))
    return matrix


def select_train_only_genes(
    gene_adata: ad.AnnData,
    labels: np.ndarray,
    train_idx: np.ndarray,
    n_genes: int,
) -> np.ndarray:
    train_matrix = _log1p_sparse(gene_adata.X[train_idx])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        scores, _ = f_classif(train_matrix, labels[train_idx])
    scores = np.nan_to_num(scores, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
    n_genes = min(int(n_genes), gene_adata.n_vars)
    selected = np.argsort(scores, kind="stable")[-n_genes:]
    return np.sort(selected.astype(np.int64))


def _to_dense_log1p(matrix) -> np.ndarray:
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    values = np.asarray(matrix, dtype=np.float32)
    return np.log1p(np.clip(values, 0.0, None)).astype(np.float32)


def _transform_splitwise(
    matrix: np.ndarray,
    transformer,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> np.ndarray:
    transformer.fit(matrix[train_idx])
    for indices in (train_idx, val_idx, test_idx):
        matrix[indices] = transformer.transform(matrix[indices]).astype(np.float32)
    return matrix


def build_fold_arrays(
    gene_path: Path,
    protein_path: Path,
    label_column: str,
    test_donor: str,
    n_genes: int,
    n_quantiles: int,
) -> tuple[dict, dict, pd.DataFrame, pd.DataFrame]:
    validation_donor = fixed_validation_donor(test_donor)
    gene_adata = ad.read_h5ad(gene_path, backed="r")
    protein_adata = ad.read_h5ad(protein_path, backed="r")
    if gene_adata.n_obs != protein_adata.n_obs or not np.array_equal(
        gene_adata.obs_names.to_numpy(),
        protein_adata.obs_names.to_numpy(),
    ):
        raise ValueError("PBMC RNA and ADT cells are not exactly aligned")
    required_columns = {label_column, "donor"}
    missing = required_columns.difference(gene_adata.obs.columns)
    if missing:
        raise ValueError(f"PBMC metadata missing columns: {sorted(missing)}")

    cell_ids = gene_adata.obs_names.to_numpy(dtype=str)
    donors = gene_adata.obs["donor"].astype(str).to_numpy()
    labels_text = gene_adata.obs[label_column].astype(str).to_numpy()
    splits = assign_nested_splits(donors, test_donor, validation_donor)
    train_idx = np.flatnonzero(splits == "train")
    val_idx = np.flatnonzero(splits == "val")
    test_idx = np.flatnonzero(splits == "test")

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels_text).astype(np.int64)
    gene_feature_idx = select_train_only_genes(
        gene_adata,
        labels,
        train_idx,
        n_genes,
    )
    selected_gene_view = gene_adata[:, gene_feature_idx].to_memory()
    protein_memory = protein_adata.to_memory()
    gene_matrix = _to_dense_log1p(selected_gene_view.X)
    protein_matrix = _to_dense_log1p(protein_memory.X)

    gene_scaler = StandardScaler(copy=False)
    protein_quantile = QuantileTransformer(
        n_quantiles=min(int(n_quantiles), len(train_idx)),
        output_distribution="normal",
        random_state=42,
        copy=False,
    )
    protein_scaler = StandardScaler(copy=False)
    gene_matrix = _transform_splitwise(
        gene_matrix,
        gene_scaler,
        train_idx,
        val_idx,
        test_idx,
    )
    protein_matrix = _transform_splitwise(
        protein_matrix,
        protein_quantile,
        train_idx,
        val_idx,
        test_idx,
    )
    protein_matrix = _transform_splitwise(
        protein_matrix,
        protein_scaler,
        train_idx,
        val_idx,
        test_idx,
    )

    arrays = {
        "gene": gene_matrix,
        "protein": protein_matrix,
        "labels": labels,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "label_encoder": label_encoder,
        "gene_names": selected_gene_view.var_names.astype(str).tolist(),
        "protein_names": protein_memory.var_names.astype(str).tolist(),
        "cell_ids": cell_ids,
        "donors": donors,
    }
    validate_nested_arrays(arrays, test_donor, validation_donor)
    support = build_label_support(labels_text, splits)
    manifest = pd.DataFrame(
        {
            "cell_id": cell_ids,
            "donor": donors,
            "label": labels_text,
            "split": splits,
            "known_to_train": np.isin(
                labels_text,
                support.loc[support["known_to_train"], "class_label"],
            ),
        }
    )
    metadata = {
        "date": DATE,
        "protocol": "nested_leave_one_donor_out",
        "test_donor": test_donor,
        "validation_donor": validation_donor,
        "training_donors": sorted(set(donors).difference({test_donor, validation_donor})),
        "label_column": label_column,
        "n_cells": int(len(labels)),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "n_genes": int(gene_matrix.shape[1]),
        "n_proteins": int(protein_matrix.shape[1]),
        "n_classes": int(len(label_encoder.classes_)),
        "natural_unknown_labels": support.loc[
            support["natural_unknown_in_test"], "class_label"
        ].astype(str).tolist(),
        "feature_selection_fit": "training donors only",
        "scalers_fit": "training donors only",
    }
    preprocessors = {
        "gene_feature_idx": gene_feature_idx,
        "gene_names": arrays["gene_names"],
        "gene_scaler": gene_scaler,
        "protein_quantile": protein_quantile,
        "protein_scaler": protein_scaler,
    }
    return arrays, {"metadata": metadata, "preprocessors": preprocessors}, support, manifest


def fold_paths(root: Path, test_donor: str, n_genes: int) -> dict[str, Path]:
    validation_donor = fixed_validation_donor(test_donor)
    stem = f"test_{test_donor}_val_{validation_donor}_g{n_genes}"
    return {
        "cache": root / f"cache/mosaic_n_v33/{stem}.npz",
        "preprocessor": root / f"cache/mosaic_n_v33/{stem}_preprocessors.joblib",
        "metadata": root / f"cache/mosaic_n_v33/{stem}_metadata.json",
        "manifest": root / f"configs/splits/pbmc_cite_seq/nested_lodo/{stem}.csv",
        "support": root / f"results/tables/mosaic_n_v33_{stem}_label_support.csv",
    }


def build_and_write_fold(args: argparse.Namespace, test_donor: str) -> dict:
    start = time.perf_counter()
    paths = fold_paths(ROOT, test_donor, args.n_genes)
    if paths["cache"].exists() and not args.force:
        arrays = load_strict_arrays_cache(paths["cache"])
        validate_nested_arrays(
            arrays,
            test_donor=test_donor,
            validation_donor=fixed_validation_donor(test_donor),
        )
        metadata = arrays.get("metadata", {})
        return {
            **metadata,
            "cache_path": str(paths["cache"].relative_to(ROOT)),
            "manifest_path": str(paths["manifest"].relative_to(ROOT)),
            "support_path": str(paths["support"].relative_to(ROOT)),
            "runtime_seconds": float(time.perf_counter() - start),
            "reused": True,
        }

    arrays, bundle, support, manifest = build_fold_arrays(
        gene_path=args.gene_path,
        protein_path=args.protein_path,
        label_column=args.label_column,
        test_donor=test_donor,
        n_genes=args.n_genes,
        n_quantiles=args.n_quantiles,
    )
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    save_strict_arrays_cache(arrays, paths["cache"], metadata=bundle["metadata"])
    joblib.dump(bundle["preprocessors"], paths["preprocessor"], compress=3)
    paths["metadata"].write_text(
        json.dumps(bundle["metadata"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    manifest.to_csv(paths["manifest"], index=False)
    support.to_csv(paths["support"], index=False)
    return {
        **bundle["metadata"],
        "cache_path": str(paths["cache"].relative_to(ROOT)),
        "manifest_path": str(paths["manifest"].relative_to(ROOT)),
        "support_path": str(paths["support"].relative_to(ROOT)),
        "runtime_seconds": float(time.perf_counter() - start),
        "reused": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gene-path", type=Path, default=Path("data/pbmc/pbmc_gene.h5ad"))
    parser.add_argument("--protein-path", type=Path, default=Path("data/pbmc/pbmc_protein.h5ad"))
    parser.add_argument("--label-column", default="celltype.l3")
    parser.add_argument("--test-donor", choices=DONOR_ORDER)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--n-genes", type=int, default=3000)
    parser.add_argument("--n-quantiles", type=int, default=1000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.all == (args.test_donor is not None):
        parser.error("choose exactly one of --all or --test-donor")
    return args


def main() -> None:
    args = parse_args()
    donors = DONOR_ORDER if args.all else [args.test_donor]
    rows = []
    for donor in donors:
        print(
            f"Building nested LODO cache: test={donor}, "
            f"val={fixed_validation_donor(donor)}",
            flush=True,
        )
        rows.append(build_and_write_fold(args, donor))
    summary = pd.DataFrame(rows)
    out_path = ROOT / "results/tables/mosaic_n_v33_nested_lodo_cache_summary_2026-07-23.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    output_path = ROOT / "output/tables/mosaic_n_v33_nested_lodo_cache_summary_2026-07-23.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
