#!/usr/bin/env python
"""Strict train-first L3 rerun for the original MOSAIC model."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.feature_selection import f_classif
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, QuantileTransformer, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path.cwd() / "code"))
from TOSICA_model_MoE import create_ultra_fast_moe_transformer  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_dense_float32(matrix) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def stratified_subsample(indices: np.ndarray, labels: np.ndarray, max_cells: int, seed: int) -> np.ndarray:
    if max_cells <= 0 or max_cells >= len(indices):
        return indices
    _, sampled = train_test_split(
        indices,
        test_size=max_cells,
        stratify=labels[indices],
        random_state=seed,
    )
    return np.sort(sampled)


def select_train_only_genes(gene_matrix: np.ndarray, labels: np.ndarray, train_idx: np.ndarray, n_genes: int) -> np.ndarray:
    scores, _ = f_classif(gene_matrix[train_idx], labels[train_idx])
    scores = np.nan_to_num(scores, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
    n_genes = min(n_genes, gene_matrix.shape[1])
    return np.argsort(scores)[-n_genes:]


def build_strict_arrays(args: argparse.Namespace) -> dict:
    gene_adata = ad.read_h5ad(args.gene_path)
    protein_adata = ad.read_h5ad(args.protein_path)
    common_cells = gene_adata.obs_names.intersection(protein_adata.obs_names)
    gene_adata = gene_adata[common_cells].copy()
    protein_adata = protein_adata[common_cells].copy()
    labels_raw = gene_adata.obs[args.label_column].astype(str).values

    label_counts = pd.Series(labels_raw).value_counts()
    valid_labels = label_counts[label_counts >= args.min_cells_per_class].index
    keep_mask = np.isin(labels_raw, valid_labels)
    gene_adata = gene_adata[keep_mask].copy()
    protein_adata = protein_adata[keep_mask].copy()
    labels_raw = labels_raw[keep_mask]

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels_raw)
    all_idx = np.arange(len(labels))
    all_idx = stratified_subsample(all_idx, labels, args.max_cells, args.seed)
    labels = labels[all_idx]
    if args.max_cells > 0:
        sampled_counts = pd.Series(labels).value_counts()
        valid_sampled_labels = sampled_counts[sampled_counts >= args.min_cells_after_subsample].index
        sampled_keep = np.isin(labels, valid_sampled_labels)
        all_idx = all_idx[sampled_keep]
        labels = labels[sampled_keep]

    train_val_idx, test_idx_local = train_test_split(
        np.arange(len(all_idx)),
        test_size=args.test_size,
        stratify=labels,
        random_state=args.seed,
    )
    train_idx_local, val_idx_local = train_test_split(
        train_val_idx,
        test_size=args.val_size / (1.0 - args.test_size),
        stratify=labels[train_val_idx],
        random_state=args.seed,
    )

    selected_global_idx = all_idx
    gene_matrix = to_dense_float32(gene_adata.X[selected_global_idx])
    protein_matrix = to_dense_float32(protein_adata.X[selected_global_idx])
    gene_matrix = np.nan_to_num(gene_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    protein_matrix = np.nan_to_num(protein_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    gene_matrix = np.log1p(np.clip(gene_matrix, 0.0, None))
    protein_matrix = np.log1p(np.clip(protein_matrix, 0.0, None))

    gene_feature_idx = select_train_only_genes(gene_matrix, labels, train_idx_local, args.n_genes)
    gene_matrix = gene_matrix[:, gene_feature_idx]

    gene_scaler = StandardScaler()
    protein_quantile = QuantileTransformer(
        n_quantiles=min(args.n_quantiles, len(train_idx_local)),
        output_distribution="normal",
        random_state=args.seed,
    )
    protein_scaler = StandardScaler()

    gene_matrix[train_idx_local] = gene_scaler.fit_transform(gene_matrix[train_idx_local])
    gene_matrix[val_idx_local] = gene_scaler.transform(gene_matrix[val_idx_local])
    gene_matrix[test_idx_local] = gene_scaler.transform(gene_matrix[test_idx_local])

    protein_matrix[train_idx_local] = protein_quantile.fit_transform(protein_matrix[train_idx_local])
    protein_matrix[val_idx_local] = protein_quantile.transform(protein_matrix[val_idx_local])
    protein_matrix[test_idx_local] = protein_quantile.transform(protein_matrix[test_idx_local])
    protein_matrix[train_idx_local] = protein_scaler.fit_transform(protein_matrix[train_idx_local])
    protein_matrix[val_idx_local] = protein_scaler.transform(protein_matrix[val_idx_local])
    protein_matrix[test_idx_local] = protein_scaler.transform(protein_matrix[test_idx_local])

    return {
        "gene": gene_matrix.astype(np.float32),
        "protein": protein_matrix.astype(np.float32),
        "labels": labels.astype(np.int64),
        "train_idx": train_idx_local,
        "val_idx": val_idx_local,
        "test_idx": test_idx_local,
        "label_encoder": label_encoder,
        "gene_names": gene_adata.var_names[gene_feature_idx].tolist(),
        "protein_names": protein_adata.var_names.tolist(),
        "cell_ids": gene_adata.obs_names[selected_global_idx].to_numpy(),
    }


def make_loader(gene: np.ndarray, protein: np.ndarray, labels: np.ndarray, indices: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(gene[indices], dtype=torch.float32),
        torch.tensor(protein[indices], dtype=torch.float32),
        torch.tensor(labels[indices], dtype=torch.long),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def compute_summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    wp, wr, wf1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    _, _, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "weighted_precision": wp,
        "weighted_recall": wr,
        "weighted_f1": wf1,
        "macro_f1": macro_f1,
    }


def safe_moe_auxiliary_loss(model: torch.nn.Module, device: torch.device) -> torch.Tensor:
    """Compute device-safe MoE load-balance/diversity loss without editing legacy model code."""

    if not hasattr(model, "moe_layers") or len(model.moe_layers) == 0:
        return torch.tensor(0.0, device=device)
    losses = []
    for moe_layer in model.moe_layers:
        gate = getattr(moe_layer, "gate", None)
        if gate is None or not hasattr(gate, "expert_usage") or not hasattr(gate, "total_tokens"):
            continue
        total_tokens = gate.total_tokens.to(device).clamp_min(1.0)
        usage_rates = gate.expert_usage.to(device) / total_tokens
        if usage_rates.numel() == 0:
            continue
        target = torch.full_like(usage_rates, 1.0 / usage_rates.numel())
        balance_loss = F.mse_loss(usage_rates, target)
        entropy = -(usage_rates * torch.log(usage_rates.clamp_min(1e-8))).sum()
        max_entropy = torch.log(torch.tensor(float(usage_rates.numel()), device=device))
        diversity_loss = (max_entropy - entropy).clamp_min(0.0)
        losses.append(balance_loss + diversity_loss)
    if not losses:
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions, labels = [], []
    for gene, protein, y in loader:
        logits = model(gene.to(device), protein.to(device))
        predictions.append(logits.argmax(dim=1).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(labels), np.concatenate(predictions)


def train_one_model(arrays: dict, args: argparse.Namespace, use_moe: bool, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = {
        "embed_dim": args.embed_dim,
        "num_experts": args.num_experts,
        "top_k": min(args.top_k, args.num_experts),
        "num_transformer_layers": args.num_transformer_layers,
        "num_moe_layers": args.num_moe_layers if use_moe else 0,
        "num_heads": args.num_heads,
        "dropout": args.dropout,
    }
    model = create_ultra_fast_moe_transformer(
        arrays["gene"].shape[1],
        arrays["protein"].shape[1],
        len(arrays["label_encoder"].classes_),
        model_config,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    train_loader = make_loader(arrays["gene"], arrays["protein"], arrays["labels"], arrays["train_idx"], args.batch_size, True)
    val_loader = make_loader(arrays["gene"], arrays["protein"], arrays["labels"], arrays["val_idx"], args.batch_size, False)
    test_loader = make_loader(arrays["gene"], arrays["protein"], arrays["labels"], arrays["test_idx"], args.batch_size, False)

    best_state = None
    best_val_macro = -1.0
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0
        for gene, protein, y in train_loader:
            gene = gene.to(device)
            protein = protein.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(gene, protein)
            loss = criterion(logits, y)
            if use_moe and args.moe_aux_weight > 0 and hasattr(model, "get_moe_auxiliary_loss"):
                loss = loss + args.moe_aux_weight * safe_moe_auxiliary_loss(model, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += float(loss.item()) * len(y)
            total += len(y)
            correct += int((logits.argmax(dim=1) == y).sum().item())
        val_true, val_pred = evaluate(model, val_loader, device)
        val_summary = compute_summary(val_true, val_pred)
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / max(total, 1),
                "train_accuracy": correct / max(total, 1),
                "val_accuracy": val_summary["accuracy"],
                "val_macro_f1": val_summary["macro_f1"],
                "val_weighted_f1": val_summary["weighted_f1"],
            }
        )
        if val_summary["macro_f1"] > best_val_macro:
            best_val_macro = val_summary["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_true, test_pred = evaluate(model, test_loader, device)
    summary = compute_summary(test_true, test_pred)
    summary.update(
        {
            "method": "Full_MOE_Gene_Protein" if use_moe else "NoMOE_Gene_Protein",
            "best_val_macro_f1": best_val_macro,
            "epochs_ran": len(history),
            "n_train": int(len(arrays["train_idx"])),
            "n_val": int(len(arrays["val_idx"])),
            "n_test": int(len(arrays["test_idx"])),
            "n_genes": int(arrays["gene"].shape[1]),
            "n_proteins": int(arrays["protein"].shape[1]),
            "n_classes": int(len(arrays["label_encoder"].classes_)),
        }
    )
    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
    pd.DataFrame([summary]).to_csv(out_dir / "results_summary.csv", index=False)
    pd.DataFrame(
        {
            "cell_id": arrays["cell_ids"][arrays["test_idx"]],
            "label": arrays["label_encoder"].classes_[test_true],
            "prediction": arrays["label_encoder"].classes_[test_pred],
        }
    ).to_csv(out_dir / "predictions.csv", index=False)
    torch.save(model.state_dict(), out_dir / "model.pt")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gene-path", type=Path, default=Path("data/pbmc/pbmc_gene.h5ad"))
    parser.add_argument("--protein-path", type=Path, default=Path("data/pbmc/pbmc_protein.h5ad"))
    parser.add_argument("--label-column", default="celltype.l3")
    parser.add_argument("--out-dir", type=Path, default=Path("results/exp_generalization/original_mosaic_strict/l3_full_vs_nomoe"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--n-genes", type=int, default=3000)
    parser.add_argument("--n-quantiles", type=int, default=1000)
    parser.add_argument("--min-cells-per-class", type=int, default=2)
    parser.add_argument("--min-cells-after-subsample", type=int, default=5)
    parser.add_argument("--max-cells", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--num-transformer-layers", type=int, default=2)
    parser.add_argument("--num-moe-layers", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--moe-aux-weight", type=float, default=0.05)
    parser.add_argument("--only", choices=["both", "moe", "nomoe"], default="both")
    args = parser.parse_args()

    start = time.perf_counter()
    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    arrays = build_strict_arrays(args)
    summaries = []
    if args.only in {"both", "moe"}:
        summaries.append(train_one_model(arrays, args, True, args.out_dir / "full_moe_gene_protein"))
    if args.only in {"both", "nomoe"}:
        summaries.append(train_one_model(arrays, args, False, args.out_dir / "nomoe_gene_protein"))
    summary_frame = pd.DataFrame(summaries)
    summary_frame["total_runtime_seconds"] = time.perf_counter() - start
    summary_frame.to_csv(args.out_dir / "comparison_summary.csv", index=False)
    with open(args.out_dir / "config.json", "w") as f:
        json.dump(
            {
                **vars(args),
                "gene_path": str(args.gene_path),
                "protein_path": str(args.protein_path),
                "strict_protocol": "split first; train-only gene f_classif selection; train-only gene/protein scalers",
            },
            f,
            indent=2,
            default=str,
        )
    print(summary_frame.to_string(index=False))


if __name__ == "__main__":
    main()
