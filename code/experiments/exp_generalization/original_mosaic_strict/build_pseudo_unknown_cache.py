from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import LabelEncoder, QuantileTransformer, StandardScaler

from run_l3_strict_mosaic import to_dense_float32

ROOT = Path(__file__).resolve().parents[3]
GENE_PATH = ROOT / "data" / "pbmc" / "pbmc_gene.h5ad"
PROTEIN_PATH = ROOT / "data" / "pbmc" / "pbmc_protein.h5ad"
SPLIT_TEMPLATE = ROOT / "configs" / "splits" / "pbmc_cite_seq" / "pseudo_unknown_{target}_seed42.csv"
AUDIT_OUTPUT = ROOT / "results" / "tables" / "pseudo_unknown_cache_audit_2026-05-17.csv"

REQUIRED_ARRAY_KEYS = [
    "gene",
    "protein",
    "labels",
    "train_idx",
    "val_idx",
    "test_known_idx",
    "test_unknown_idx",
    "cell_ids",
    "label_classes",
    "gene_names",
    "protein_names",
    "known_unknown",
    "metadata_json",
]

AUDIT_COLUMNS = [
    "target_label",
    "cache_path",
    "n_cells",
    "n_train",
    "n_val",
    "n_test_known",
    "n_test_unknown",
    "n_classes",
    "n_genes",
    "n_proteins",
    "max_cells",
]


def split_path_for(root: Path, target_label: str) -> Path:
    path_text = SPLIT_TEMPLATE.relative_to(ROOT).as_posix().format(target=target_label)
    return root / path_text


def sample_active_split(split: pd.DataFrame, target_label: str, max_cells: int, seed: int) -> pd.DataFrame:
    active = split[split["split"].isin(["train", "val", "test"])].copy()
    active = active[active["known_unknown"].isin(["known", "unknown"])].copy()
    if max_cells <= 0 or len(active) <= max_cells:
        return active.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    unknown = active[active["known_unknown"] == "unknown"].copy()
    known = active[active["known_unknown"] == "known"].copy()
    unknown_keep_n = min(len(unknown), max(1, max_cells // 5))
    if len(unknown) > unknown_keep_n:
        unknown = unknown.iloc[np.sort(rng.choice(len(unknown), size=unknown_keep_n, replace=False))].copy()
    remaining = max_cells - len(unknown)
    split_fracs = {"train": 0.72, "val": 0.10, "test": 0.18}
    known_parts = []
    for split_name, frac in split_fracs.items():
        part = known[known["split"] == split_name]
        take = min(len(part), max(1, int(round(remaining * frac))))
        if len(part) > take:
            part = part.iloc[np.sort(rng.choice(len(part), size=take, replace=False))].copy()
        known_parts.append(part)
    sampled = pd.concat(known_parts + [unknown], ignore_index=True)
    if len(sampled) > max_cells:
        sampled = sampled.sample(n=max_cells, random_state=seed).copy()
    return sampled.reset_index(drop=True)


def select_train_only_genes(gene_matrix: np.ndarray, labels: np.ndarray, train_idx: np.ndarray, n_genes: int) -> np.ndarray:
    train_labels = labels[train_idx]
    scores, _ = f_classif(gene_matrix[train_idx], train_labels)
    scores = np.nan_to_num(scores, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
    n_genes = min(int(n_genes), gene_matrix.shape[1])
    return np.argsort(scores)[-n_genes:]


def build_pseudo_unknown_arrays(
    root: Path = ROOT,
    target_label: str = "gdT_2",
    n_genes: int = 200,
    max_cells: int = 12000,
    seed: int = 42,
    n_quantiles: int = 1000,
) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    split_path = split_path_for(root, target_label)
    if not split_path.exists():
        raise FileNotFoundError(split_path)
    split = pd.read_csv(split_path)
    active = sample_active_split(split, target_label=target_label, max_cells=max_cells, seed=seed)
    cell_ids = np.asarray(active["cell_id"].astype(str).tolist(), dtype=str)

    gene_path = root / GENE_PATH.relative_to(ROOT)
    protein_path = root / PROTEIN_PATH.relative_to(ROOT)
    gene_adata = ad.read_h5ad(gene_path)
    protein_adata = ad.read_h5ad(protein_path)
    try:
        gene_adata = gene_adata[cell_ids].copy()
        protein_adata = protein_adata[cell_ids].copy()
        gene_matrix = to_dense_float32(gene_adata.X)
        protein_matrix = to_dense_float32(protein_adata.X)
        gene_names_all = np.asarray(gene_adata.var_names.astype(str))
        protein_names = np.asarray(protein_adata.var_names.astype(str))
    finally:
        try:
            gene_adata.file.close()
        except Exception:
            pass
        try:
            protein_adata.file.close()
        except Exception:
            pass

    gene_matrix = np.nan_to_num(gene_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    protein_matrix = np.nan_to_num(protein_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    gene_matrix = np.log1p(np.clip(gene_matrix, 0.0, None))
    protein_matrix = np.log1p(np.clip(protein_matrix, 0.0, None))

    known_mask = active["known_unknown"].astype(str).to_numpy() == "known"
    train_idx = np.where((active["split"].astype(str).to_numpy() == "train") & known_mask)[0].astype(np.int64)
    val_idx = np.where((active["split"].astype(str).to_numpy() == "val") & known_mask)[0].astype(np.int64)
    test_known_idx = np.where((active["split"].astype(str).to_numpy() == "test") & known_mask)[0].astype(np.int64)
    test_unknown_idx = np.where(active["known_unknown"].astype(str).to_numpy() == "unknown")[0].astype(np.int64)

    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_known_idx) == 0 or len(test_unknown_idx) == 0:
        raise ValueError("train/val/test_known/test_unknown must all be non-empty")
    train_labels_raw = active.iloc[train_idx]["label"].astype(str).to_numpy()
    label_encoder = LabelEncoder()
    label_encoder.fit(train_labels_raw)
    target_in_classes = target_label in set(label_encoder.classes_.astype(str))
    if target_in_classes:
        raise ValueError(f"target label leaked into known label classes: {target_label}")

    labels = np.full(len(active), -1, dtype=np.int64)
    known_indices = np.where(known_mask)[0]
    known_labels_raw = active.iloc[known_indices]["label"].astype(str).to_numpy()
    unseen_known = sorted(set(known_labels_raw).difference(set(label_encoder.classes_.astype(str))))
    if unseen_known:
        raise ValueError("known labels absent from train classes: " + ",".join(unseen_known[:10]))
    labels[known_indices] = label_encoder.transform(known_labels_raw)

    gene_feature_idx = select_train_only_genes(gene_matrix, labels, train_idx, n_genes)
    gene_matrix = gene_matrix[:, gene_feature_idx]
    gene_names = gene_names_all[gene_feature_idx]

    gene_scaler = StandardScaler()
    protein_quantile = QuantileTransformer(
        n_quantiles=min(int(n_quantiles), len(train_idx)),
        output_distribution="normal",
        random_state=seed,
    )
    protein_scaler = StandardScaler()
    transform_indices = np.concatenate([train_idx, val_idx, test_known_idx, test_unknown_idx]).astype(np.int64)
    gene_matrix[train_idx] = gene_scaler.fit_transform(gene_matrix[train_idx])
    gene_matrix[np.setdiff1d(transform_indices, train_idx)] = gene_scaler.transform(gene_matrix[np.setdiff1d(transform_indices, train_idx)])
    protein_matrix[train_idx] = protein_quantile.fit_transform(protein_matrix[train_idx])
    protein_matrix[np.setdiff1d(transform_indices, train_idx)] = protein_quantile.transform(protein_matrix[np.setdiff1d(transform_indices, train_idx)])
    protein_matrix[train_idx] = protein_scaler.fit_transform(protein_matrix[train_idx])
    protein_matrix[np.setdiff1d(transform_indices, train_idx)] = protein_scaler.transform(protein_matrix[np.setdiff1d(transform_indices, train_idx)])

    metadata = {
        "target_label": target_label,
        "seed": int(seed),
        "n_genes": int(gene_matrix.shape[1]),
        "max_cells": int(max_cells),
        "source_split": split_path.relative_to(root).as_posix(),
        "unknown_contract": "labels=-1 for test_unknown_idx; label_classes exclude target_label",
    }
    arrays: Dict[str, np.ndarray] = {
        "gene": gene_matrix.astype(np.float32),
        "protein": protein_matrix.astype(np.float32),
        "labels": labels.astype(np.int64),
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_known_idx": test_known_idx,
        "test_unknown_idx": test_unknown_idx,
        "cell_ids": np.asarray(cell_ids.tolist(), dtype=str),
        "label_classes": np.asarray(label_encoder.classes_.astype(str).tolist(), dtype=str),
        "gene_names": np.asarray(gene_names.astype(str).tolist(), dtype=str),
        "protein_names": np.asarray(protein_names.astype(str).tolist(), dtype=str),
        "known_unknown": np.asarray(active["known_unknown"].astype(str).tolist(), dtype=str),
        "metadata_json": np.asarray(json.dumps(metadata, ensure_ascii=False), dtype=str),
    }
    return arrays, metadata


def save_arrays(arrays: Dict[str, np.ndarray], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)


def audit_row(target_label: str, output_path: Path, arrays: Dict[str, np.ndarray], metadata: Dict[str, object]) -> Dict[str, str]:
    return {
        "target_label": target_label,
        "cache_path": output_path.relative_to(ROOT).as_posix(),
        "n_cells": str(int(arrays["gene"].shape[0])),
        "n_train": str(int(len(arrays["train_idx"]))),
        "n_val": str(int(len(arrays["val_idx"]))),
        "n_test_known": str(int(len(arrays["test_known_idx"]))),
        "n_test_unknown": str(int(len(arrays["test_unknown_idx"]))),
        "n_classes": str(int(len(arrays["label_classes"]))),
        "n_genes": str(int(arrays["gene"].shape[1])),
        "n_proteins": str(int(arrays["protein"].shape[1])),
        "max_cells": str(metadata["max_cells"]),
    }


def merge_audit_rows(existing_rows: List[Dict[str, str]], incoming_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    by_target: Dict[str, Dict[str, str]] = {}
    for row in existing_rows:
        target = str(row.get("target_label", ""))
        if target:
            by_target[target] = dict(row)
    for row in incoming_rows:
        target = str(row.get("target_label", ""))
        if target:
            by_target[target] = dict(row)
    return [by_target[target] for target in sorted(by_target)]


def write_audit(rows: List[Dict[str, str]], output_path: Path = AUDIT_OUTPUT) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing_rows: List[Dict[str, str]] = []
    if output_path.exists():
        with output_path.open("r", newline="", encoding="utf-8") as handle:
            existing_rows = list(csv.DictReader(handle))
    merged_rows = merge_audit_rows(existing_rows, rows)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_COLUMNS)
        writer.writeheader()
        for row in merged_rows:
            writer.writerow({column: row.get(column, "") for column in AUDIT_COLUMNS})
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-label", default="gdT_2")
    parser.add_argument("--n-genes", type=int, default=200)
    parser.add_argument("--max-cells", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-quantiles", type=int, default=1000)
    parser.add_argument("--output-suffix", default="smoke_g200")
    args = parser.parse_args()

    arrays, metadata = build_pseudo_unknown_arrays(
        root=ROOT,
        target_label=args.target_label,
        n_genes=args.n_genes,
        max_cells=args.max_cells,
        seed=args.seed,
        n_quantiles=args.n_quantiles,
    )
    safe_target = args.target_label.replace(" ", "_")
    output_path = ROOT / "cache" / "original_mosaic_strict" / f"pseudo_unknown_{safe_target}_{args.output_suffix}_seed{args.seed}_arrays.npz"
    save_arrays(arrays, output_path)
    audit_path = write_audit([audit_row(args.target_label, output_path, arrays, metadata)])
    print("pseudo-unknown smoke cache written")
    print(f"cache: {output_path}")
    print(f"audit: {audit_path}")
    n_cells = arrays["gene"].shape[0]
    n_train = len(arrays["train_idx"])
    n_val = len(arrays["val_idx"])
    n_test_known = len(arrays["test_known_idx"])
    n_test_unknown = len(arrays["test_unknown_idx"])
    n_classes = len(arrays["label_classes"])
    print(f"cells={n_cells} train={n_train} val={n_val} test_known={n_test_known} test_unknown={n_test_unknown} classes={n_classes}")


if __name__ == "__main__":
    main()
