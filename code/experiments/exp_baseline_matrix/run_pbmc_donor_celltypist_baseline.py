#!/usr/bin/env python
"""Run CellTypist on the primary PBMC nested donor caches."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support


ROOT = Path(__file__).resolve().parents[2]
STRICT_DIR = ROOT / "experiments/exp_generalization/original_mosaic_strict"
if str(STRICT_DIR) not in sys.path:
    sys.path.insert(0, str(STRICT_DIR))

from strict_array_cache import load_strict_arrays_cache  # noqa: E402


DATE = "2026-07-29"
DONORS = [f"P{i}" for i in range(1, 9)]
METHOD = "CellTypist"


def relative_path_text(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def as_str_array(values) -> np.ndarray:
    return np.asarray(values).astype(str)


def next_val_donor(donor: str) -> str:
    return f"P{(int(donor[1:]) % 8) + 1}"


def cache_path_for_donor(donor: str) -> Path:
    return ROOT / f"cache/mosaic_n_v33/test_{donor}_val_{next_val_donor(donor)}_g3000.npz"


def extract_celltypist_predictions(result) -> tuple[np.ndarray, np.ndarray | None]:
    labels = result.predicted_labels
    if isinstance(labels, pd.DataFrame):
        if "predicted_labels" in labels.columns:
            y_pred = labels["predicted_labels"].astype(str).to_numpy()
        else:
            y_pred = labels.iloc[:, 0].astype(str).to_numpy()
    else:
        y_pred = pd.Series(labels).astype(str).to_numpy()

    confidence = None
    probability = getattr(result, "probability_matrix", None)
    if probability is not None:
        confidence = pd.DataFrame(probability).max(axis=1).to_numpy(dtype=float)
    return y_pred, confidence


def annotate_standardized_matrix(model, x_test: np.ndarray, gene_names: np.ndarray, cell_ids: np.ndarray):
    from celltypist.classifier import Classifier

    clf = Classifier(filename="", model=model)
    clf.indata = x_test
    clf.indata_genes = gene_names
    clf.indata_names = cell_ids
    clf.adata = None
    return clf.celltype()


def metric_record(donor: str, y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], runtime: float, n_train_fit: int, n_genes: int) -> dict:
    return {
        "method": METHOD,
        "method_type": "published_single_cell_annotation",
        "test_donor": donor,
        "accuracy": accuracy_score(y_true, y_pred),
        "weighted_f1": f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "n_train_fit": int(n_train_fit),
        "n_test": int(len(y_true)),
        "n_genes": int(n_genes),
        "runtime_seconds": float(runtime),
    }


def per_class_frame(donor: str, y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> pd.DataFrame:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    return pd.DataFrame(
        {
            "method": METHOD,
            "test_donor": donor,
            "label": labels,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )


def summarize(run_level: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, method_frame in run_level.groupby("method", sort=False):
        for metric in ["accuracy", "weighted_f1", "macro_f1", "balanced_accuracy"]:
            values = method_frame[metric].to_numpy(dtype=float)
            mean = float(values.mean())
            sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            margin = float(stats.t.ppf(0.975, len(values) - 1) * sd / np.sqrt(len(values))) if len(values) > 1 else 0.0
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "n_donors": int(len(values)),
                    "mean": mean,
                    "sd_across_donors": sd,
                    "ci95_low": mean - margin,
                    "ci95_high": mean + margin,
                    "ci95_margin": margin,
                    "worst_donor_value": float(values.min()),
                    "best_donor_value": float(values.max()),
                }
            )
    return pd.DataFrame(rows)


def build_config(
    donors: list[str],
    max_iter: int,
    n_jobs: int,
    runtime_seconds: float,
    celltypist_version: str,
) -> dict:
    return {
        "date": DATE,
        "method": METHOD,
        "method_type": "published_single_cell_annotation",
        "protocol": "primary nested leave-one-donor-out PBMC L3",
        "donors": donors,
        "max_train_per_class": 0,
        "cache_pattern": "cache/mosaic_n_v33/test_<donor>_val_<next_donor>_g3000.npz",
        "input_modality": "RNA-only",
        "input_features": "primary train-only selected and standardized RNA features",
        "celltypist_train_check_expression": False,
        "celltypist_annotation_interface": "Classifier(filename='', model=model).celltype() on standardized matrix",
        "celltypist_annotate_majority_voting": False,
        "celltypist_max_iter": int(max_iter),
        "n_jobs": int(n_jobs),
        "test_label_tuning": False,
        "statistical_unit": "held-out donor",
        "celltypist_version": celltypist_version,
        "runtime_seconds": float(runtime_seconds),
        "protocol_caveat": (
            "Published CellTypist package trained per fold on primary nested-donor RNA features; "
            "full training donors; full held-out donor test; package predictor called directly on "
            "standardized matrices because the primary cache stores train-only standardized RNA tensors; "
            "no test-label tuning."
        ),
    }


def write_environment(out_dir: Path, celltypist_version: str) -> None:
    packages = {"celltypist": celltypist_version}
    for module_name in ["numpy", "pandas", "sklearn", "anndata"]:
        try:
            module = __import__(module_name)
            packages[module_name] = getattr(module, "__version__", "unknown")
        except ImportError:
            packages[module_name] = "not_installed"
    environment = {
        "date": DATE,
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "conda_prefix": os.environ.get("CONDA_PREFIX", ""),
        "packages": packages,
    }
    (out_dir / "environment.json").write_text(json.dumps(environment, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    import celltypist

    start = time.perf_counter()
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    celltypist_version = getattr(celltypist, "__version__", "unknown")
    write_environment(out_dir, celltypist_version)

    run_rows = []
    pred_frames = []
    per_class_frames = []
    missing = []

    for donor in args.donors:
        cache = cache_path_for_donor(donor)
        if not cache.exists():
            missing.append(relative_path_text(cache))
            continue
        arrays = load_strict_arrays_cache(cache)
        class_names = as_str_array(arrays["label_encoder"].classes_)
        label_names = class_names.tolist()
        train_idx = arrays["train_idx"]
        test_idx = arrays["test_idx"]
        gene_names = as_str_array(arrays["gene_names"])
        x_train = np.asarray(arrays["gene"][train_idx], dtype=np.float32)
        x_test = np.asarray(arrays["gene"][test_idx], dtype=np.float32)
        y_train = class_names[arrays["labels"][train_idx]]
        y_test = class_names[arrays["labels"][test_idx]]

        method_start = time.perf_counter()
        print(f"start donor={donor} method={METHOD} n_train_fit={len(train_idx)}", flush=True)
        model = celltypist.train(
            X=x_train,
            labels=y_train,
            genes=gene_names.tolist(),
            check_expression=False,
            C=args.c_value,
            max_iter=args.max_iter,
            n_jobs=args.n_jobs,
            use_SGD=args.use_sgd,
            alpha=args.alpha,
            balance_cell_type=args.balance_cell_type,
        )
        test_cell_ids = as_str_array(arrays["cell_ids"])[test_idx]
        result = annotate_standardized_matrix(model, x_test, gene_names, test_cell_ids)
        y_pred, confidence = extract_celltypist_predictions(result)
        runtime = float(time.perf_counter() - method_start)

        run_rows.append(metric_record(donor, y_test, y_pred, label_names, runtime, len(train_idx), len(gene_names)))
        per_class_frames.append(per_class_frame(donor, y_test, y_pred, label_names))
        pred_frame = pd.DataFrame(
            {
                "method": METHOD,
                "test_donor": donor,
                "cell_id": arrays["cell_ids"][test_idx],
                "true_label": y_test,
                "pred_label": y_pred,
            }
        )
        if confidence is not None:
            pred_frame["confidence"] = confidence
        pred_frames.append(pred_frame)
        pd.DataFrame(run_rows).to_csv(out_dir / "run_level_metrics.csv", index=False)
        print(f"done donor={donor} method={METHOD} acc={run_rows[-1]['accuracy']:.4f}", flush=True)

    if args.require_complete and missing:
        raise FileNotFoundError(f"missing caches: {missing}")
    if not run_rows:
        raise RuntimeError("no CellTypist run rows were generated")

    run_level = pd.DataFrame(run_rows)
    summary = summarize(run_level)
    run_level.to_csv(out_dir / "run_level_metrics.csv", index=False)
    summary.to_csv(out_dir / "results_summary.csv", index=False)
    summary.to_csv(out_dir / "donor_method_summary.csv", index=False)
    pd.concat(per_class_frames, ignore_index=True).to_csv(out_dir / "per_class_metrics.csv", index=False)
    pd.concat(pred_frames, ignore_index=True).to_csv(out_dir / "predictions.csv.gz", index=False, compression="gzip")
    config = build_config(args.donors, args.max_iter, args.n_jobs, time.perf_counter() - start, celltypist_version)
    config.update({"c_value": args.c_value, "use_sgd": args.use_sgd, "alpha": args.alpha, "balance_cell_type": args.balance_cell_type})
    (out_dir / "run_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    artifacts = []
    for path in sorted(out_dir.iterdir()):
        if path.name != "artifact_index.csv":
            artifacts.append({"artifact": path.name, "bytes": path.stat().st_size})
    pd.DataFrame(artifacts).to_csv(out_dir / "artifact_index.csv", index=False)
    for base in (ROOT / "results/tables", ROOT / "output/tables"):
        base.mkdir(parents=True, exist_ok=True)
        summary.to_csv(base / f"mosaic_n_v42_celltypist_pbmc_nested_donor_{DATE}.csv", index=False)
    print(summary.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results/exp_baseline_matrix/v42_pbmc_donor_celltypist_full_training")
    parser.add_argument("--donors", nargs="+", default=DONORS, choices=DONORS)
    parser.add_argument("--c-value", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--use-sgd", action="store_true")
    parser.add_argument("--alpha", type=float, default=1e-4)
    parser.add_argument("--balance-cell-type", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
