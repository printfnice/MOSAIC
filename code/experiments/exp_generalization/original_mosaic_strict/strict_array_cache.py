#!/usr/bin/env python
"""Cache helpers for strict train-first original MOSAIC arrays."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.preprocessing import LabelEncoder


def _as_str_array(values) -> np.ndarray:
    return np.asarray(list(values), dtype=str)


def save_strict_arrays_cache(arrays: dict, cache_path: Path, metadata: dict | None = None) -> None:
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    label_classes = _as_str_array(arrays["label_encoder"].classes_)
    payload = {
        "gene": arrays["gene"],
        "protein": arrays["protein"],
        "labels": arrays["labels"],
        "train_idx": arrays["train_idx"],
        "val_idx": arrays["val_idx"],
        "test_idx": arrays["test_idx"],
        "cell_ids": _as_str_array(arrays["cell_ids"]),
        "gene_names": _as_str_array(arrays["gene_names"]),
        "protein_names": _as_str_array(arrays["protein_names"]),
        "label_classes": label_classes,
        "metadata_json": np.asarray(json.dumps(metadata or {}, ensure_ascii=False), dtype=str),
    }
    if "donors" in arrays:
        payload["donors"] = _as_str_array(arrays["donors"])
    np.savez_compressed(cache_path, **payload)


def load_strict_arrays_cache(cache_path: Path) -> dict:
    cache_path = Path(cache_path)
    with np.load(cache_path, allow_pickle=False) as data:
        label_encoder = LabelEncoder()
        label_encoder.classes_ = data["label_classes"].astype(str)
        metadata_raw = str(data["metadata_json"].item())
        arrays = {
            "gene": data["gene"].astype(np.float32),
            "protein": data["protein"].astype(np.float32),
            "labels": data["labels"].astype(np.int64),
            "train_idx": data["train_idx"].astype(np.int64),
            "val_idx": data["val_idx"].astype(np.int64),
            "test_idx": data["test_idx"].astype(np.int64),
            "cell_ids": data["cell_ids"].astype(str),
            "gene_names": data["gene_names"].astype(str).tolist(),
            "protein_names": data["protein_names"].astype(str).tolist(),
            "label_encoder": label_encoder,
            "metadata": json.loads(metadata_raw) if metadata_raw else {},
        }
        if "donors" in data.files:
            arrays["donors"] = data["donors"].astype(str)
        return arrays


def build_cache_metadata(args) -> dict:
    fields = [
        "gene_path",
        "protein_path",
        "label_column",
        "seed",
        "test_size",
        "val_size",
        "n_genes",
        "n_quantiles",
        "min_cells_per_class",
        "min_cells_after_subsample",
        "max_cells",
    ]
    metadata = {}
    for field in fields:
        if hasattr(args, field):
            value = getattr(args, field)
            metadata[field] = str(value) if isinstance(value, Path) else value
    return metadata


def json_ready_args(args) -> dict:
    ready = {}
    for key, value in vars(args).items():
        ready[key] = str(value) if isinstance(value, Path) else value
    return ready


def build_or_load_strict_arrays(args, builder: Callable[[object], dict]) -> dict:
    cache_path = getattr(args, "cache_path", None)
    if cache_path is None:
        return builder(args)
    cache_path = Path(cache_path)
    if cache_path.exists():
        print(f"Loading strict arrays cache: {cache_path}")
        return load_strict_arrays_cache(cache_path)
    arrays = builder(args)
    save_strict_arrays_cache(arrays, cache_path, metadata=build_cache_metadata(args))
    print(f"Saved strict arrays cache: {cache_path}")
    return arrays
