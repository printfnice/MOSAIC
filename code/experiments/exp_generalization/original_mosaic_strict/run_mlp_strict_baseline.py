#!/usr/bin/env python
"""Strict train-first MLP baseline for original MOSAIC experiments."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from run_l3_strict_mosaic import build_strict_arrays, set_seed
from strict_array_cache import build_or_load_strict_arrays, json_ready_args


def parse_hidden_dims(value: str) -> list[int]:
    dims = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not dims:
        raise ValueError("--hidden-dims must contain at least one integer")
    return dims


def build_feature_matrix(arrays: dict, indices: np.ndarray, modality: str) -> np.ndarray:
    if modality == "gene":
        return arrays["gene"][indices].astype(np.float32)
    if modality == "protein":
        return arrays["protein"][indices].astype(np.float32)
    if modality == "both":
        return np.concatenate([arrays["gene"][indices], arrays["protein"][indices]], axis=1).astype(np.float32)
    raise ValueError(f"Unsupported modality: {modality}")


def summarize_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    wp, wr, wf1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    _, _, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_precision": float(wp),
        "weighted_recall": float(wr),
        "weighted_f1": float(wf1),
        "macro_f1": float(macro_f1),
    }


def compute_group_sample_weights(groups: np.ndarray) -> np.ndarray:
    groups = np.asarray(groups)
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    return (1.0 / counts[inverse]).astype(np.float64)


def coral_domain_loss(features: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
    unique_groups = torch.unique(groups)
    group_stats = []
    for group in unique_groups:
        group_features = features[groups == group]
        if group_features.shape[0] < 2:
            continue
        mean = group_features.mean(dim=0)
        centered = group_features - mean
        cov = centered.T.matmul(centered) / max(group_features.shape[0] - 1, 1)
        group_stats.append((mean, cov))
    if len(group_stats) < 2:
        return features.new_zeros(())

    loss = features.new_zeros(())
    n_pairs = 0
    feature_dim = max(features.shape[1], 1)
    for i in range(len(group_stats)):
        for j in range(i + 1, len(group_stats)):
            mean_i, cov_i = group_stats[i]
            mean_j, cov_j = group_stats[j]
            mean_loss = torch.mean((mean_i - mean_j) ** 2)
            cov_loss = torch.mean((cov_i - cov_j) ** 2) / (4.0 * feature_dim * feature_dim)
            loss = loss + mean_loss + cov_loss
            n_pairs += 1
    return loss / max(n_pairs, 1)


class StrictMLP(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], num_classes: int, dropout: float) -> None:
        super().__init__()
        layers: list[torch.nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    torch.nn.Linear(previous_dim, hidden_dim),
                    torch.nn.BatchNorm1d(hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(dropout),
                ]
            )
            previous_dim = hidden_dim
        layers.append(torch.nn.Linear(previous_dim, num_classes))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        for layer in list(self.network.children())[:-1]:
            x = layer(x)
        return x

    def forward_with_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.extract_features(x)
        classifier = list(self.network.children())[-1]
        return classifier(features), features


def average_state_dicts(states: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not states:
        raise ValueError("states must not be empty")
    averaged = {}
    for key in states[0].keys():
        values = [state[key] for state in states]
        if torch.is_floating_point(values[0]):
            averaged[key] = torch.stack([value.float() for value in values], dim=0).mean(dim=0).to(dtype=values[0].dtype)
        else:
            averaged[key] = values[-1].clone()
    return averaged


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int = 42,
    sample_weights: np.ndarray | None = None,
    groups: np.ndarray | None = None,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    tensors = [torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)]
    if groups is not None:
        tensors.append(torch.tensor(groups, dtype=torch.long))
    dataset = TensorDataset(*tensors)
    if sample_weights is not None:
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
            generator=generator,
        )
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=False)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False, generator=generator if shuffle else None)


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    predictions, labels = [], []
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += float(loss.item()) * len(y)
        total += len(y)
        predictions.append(logits.argmax(dim=1).cpu().numpy())
        labels.append(y.cpu().numpy())
    return np.concatenate(labels), np.concatenate(predictions), total_loss / max(total, 1)


@torch.no_grad()
def evaluate_with_probabilities(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    model.eval()
    predictions, labels, probabilities = [], [], []
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += float(loss.item()) * len(y)
        total += len(y)
        predictions.append(logits.argmax(dim=1).cpu().numpy())
        labels.append(y.cpu().numpy())
        probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
    return (
        np.concatenate(labels),
        np.concatenate(predictions),
        np.concatenate(probabilities),
        total_loss / max(total, 1),
    )


def make_probability_frame(
    cell_ids: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    class_names: np.ndarray,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "cell_id": cell_ids,
            "label": class_names[y_true],
            "prediction": class_names[y_pred],
        }
    )
    for class_index, class_name in enumerate(class_names):
        frame[f"prob::{class_name}"] = probabilities[:, class_index]
    return frame


def train_mlp(arrays: dict, args: argparse.Namespace) -> tuple[StrictMLP, list[dict], dict]:
    hidden_dims = parse_hidden_dims(args.hidden_dims)
    x_train = build_feature_matrix(arrays, arrays["train_idx"], args.modality)
    y_train = arrays["labels"][arrays["train_idx"]]
    x_val = build_feature_matrix(arrays, arrays["val_idx"], args.modality)
    y_val = arrays["labels"][arrays["val_idx"]]
    sample_weights = None
    if args.balance_by == "donor":
        if "donors" not in arrays:
            raise ValueError("--balance-by donor requires a cache containing arrays['donors']")
        sample_weights = compute_group_sample_weights(arrays["donors"][arrays["train_idx"]])
    elif args.balance_by == "class":
        sample_weights = compute_group_sample_weights(y_train)
    train_groups = None
    if args.domain_reg == "coral":
        if "donors" not in arrays:
            raise ValueError("--domain-reg coral requires a cache containing arrays['donors']")
        _, train_groups = np.unique(arrays["donors"][arrays["train_idx"]], return_inverse=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = StrictMLP(x_train.shape[1], hidden_dims, len(arrays["label_encoder"].classes_), args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    train_loader = make_loader(
        x_train,
        y_train,
        args.batch_size,
        shuffle=True,
        seed=args.seed,
        sample_weights=sample_weights,
        groups=train_groups,
    )
    val_loader = make_loader(x_val, y_val, args.batch_size, shuffle=False, seed=args.seed)

    best_state = None
    best_val_macro = -1.0
    best_epoch = 0
    stale_epochs = 0
    swa_states = []
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0
        for batch in train_loader:
            if len(batch) == 3:
                x, y, groups = batch
                groups = groups.to(device)
            else:
                x, y = batch
                groups = None
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            if args.domain_reg == "coral":
                logits, features = model.forward_with_features(x)
            else:
                logits = model(x)
                features = None
            loss = criterion(logits, y)
            if args.domain_reg == "coral" and features is not None and groups is not None:
                loss = loss + args.domain_lambda * coral_domain_loss(features, groups)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += float(loss.item()) * len(y)
            total += len(y)
            correct += int((logits.argmax(dim=1) == y).sum().item())

        val_true, val_pred, val_loss = evaluate(model, val_loader, device)
        val_summary = summarize_predictions(val_true, val_pred)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(total, 1),
            "train_accuracy": correct / max(total, 1),
            "val_loss": val_loss,
            "val_accuracy": val_summary["accuracy"],
            "val_weighted_f1": val_summary["weighted_f1"],
            "val_macro_f1": val_summary["macro_f1"],
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} train_loss={row['train_loss']:.4f} "
            f"val_macro_f1={row['val_macro_f1']:.4f} val_weighted_f1={row['val_weighted_f1']:.4f}",
            flush=True,
        )
        if val_summary["macro_f1"] > best_val_macro:
            best_val_macro = val_summary["macro_f1"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if args.swa and epoch >= args.swa_start_epoch:
            swa_states.append({key: value.detach().cpu().clone() for key, value in model.state_dict().items()})
            if len(swa_states) > args.swa_max_checkpoints:
                swa_states.pop(0)
        if stale_epochs >= args.patience:
            break

    selected_model = "best_val"
    if args.swa and swa_states:
        model.load_state_dict(average_state_dicts(swa_states))
        selected_model = "swa"
    elif best_state is not None:
        model.load_state_dict(best_state)
    return model, history, {
        "best_val_macro_f1": float(best_val_macro),
        "best_epoch": int(best_epoch),
        "device": str(device),
        "selected_model": selected_model,
        "swa_n_checkpoints": int(len(swa_states)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gene-path", type=Path, default=Path("data/pbmc/pbmc_gene.h5ad"))
    parser.add_argument("--protein-path", type=Path, default=Path("data/pbmc/pbmc_protein.h5ad"))
    parser.add_argument("--label-column", default="celltype.l3")
    parser.add_argument("--out-dir", type=Path, default=Path("results/exp_generalization/original_mosaic_strict/l3_mlp"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--n-genes", type=int, default=3000)
    parser.add_argument("--n-quantiles", type=int, default=1000)
    parser.add_argument("--min-cells-per-class", type=int, default=2)
    parser.add_argument("--min-cells-after-subsample", type=int, default=5)
    parser.add_argument("--max-cells", type=int, default=100000)
    parser.add_argument("--modality", choices=["both", "gene", "protein"], default="both")
    parser.add_argument("--hidden-dims", default="512,128")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--swa", action="store_true")
    parser.add_argument("--swa-start-epoch", type=int, default=25)
    parser.add_argument("--swa-max-checkpoints", type=int, default=12)
    parser.add_argument("--balance-by", choices=["none", "donor", "class"], default="none")
    parser.add_argument("--domain-reg", choices=["none", "coral"], default="none")
    parser.add_argument("--domain-lambda", type=float, default=0.0)
    parser.add_argument("--cache-path", type=Path, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--save-train-probabilities", action="store_true")
    args = parser.parse_args()

    start = time.perf_counter()
    random.seed(args.seed)
    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    arrays = build_or_load_strict_arrays(args, build_strict_arrays)
    model, history, train_info = train_mlp(arrays, args)

    x_test = build_feature_matrix(arrays, arrays["test_idx"], args.modality)
    y_test = arrays["labels"][arrays["test_idx"]]
    x_val = build_feature_matrix(arrays, arrays["val_idx"], args.modality)
    y_val = arrays["labels"][arrays["val_idx"]]
    x_train = build_feature_matrix(arrays, arrays["train_idx"], args.modality)
    y_train = arrays["labels"][arrays["train_idx"]]
    train_loader = make_loader(x_train, y_train, args.batch_size, shuffle=False, seed=args.seed)
    val_loader = make_loader(x_val, y_val, args.batch_size, shuffle=False, seed=args.seed)
    test_loader = make_loader(x_test, y_test, args.batch_size, shuffle=False, seed=args.seed)
    device = torch.device(train_info["device"])
    model = model.to(device)
    if args.save_train_probabilities:
        train_true, train_pred, train_prob, _ = evaluate_with_probabilities(model, train_loader, device)
    val_true, val_pred, val_prob, _ = evaluate_with_probabilities(model, val_loader, device)
    y_true, y_pred, test_prob, test_loss = evaluate_with_probabilities(model, test_loader, device)
    test_summary = summarize_predictions(y_true, y_pred)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "train_info": train_info,
            "label_classes": arrays["label_encoder"].classes_.tolist(),
            "args": json_ready_args(args),
        },
        args.out_dir / "model.pt",
    )
    pd.DataFrame(history).to_csv(args.out_dir / "training_history.csv", index=False)
    pd.DataFrame(
        {
            "cell_id": arrays["cell_ids"][arrays["test_idx"]],
            "label": arrays["label_encoder"].inverse_transform(y_true),
            "prediction": arrays["label_encoder"].inverse_transform(y_pred),
        }
    ).to_csv(args.out_dir / "predictions.csv", index=False)
    pd.DataFrame(
        {
            "cell_id": arrays["cell_ids"][arrays["val_idx"]],
            "label": arrays["label_encoder"].inverse_transform(val_true),
            "prediction": arrays["label_encoder"].inverse_transform(val_pred),
        }
    ).to_csv(args.out_dir / "predictions_val.csv", index=False)
    class_names = arrays["label_encoder"].classes_
    if args.save_train_probabilities:
        make_probability_frame(
            arrays["cell_ids"][arrays["train_idx"]],
            train_true,
            train_pred,
            train_prob,
            class_names,
        ).to_csv(args.out_dir / "probabilities_train.csv", index=False)
    make_probability_frame(
        arrays["cell_ids"][arrays["val_idx"]],
        val_true,
        val_pred,
        val_prob,
        class_names,
    ).to_csv(args.out_dir / "probabilities_val.csv", index=False)
    make_probability_frame(
        arrays["cell_ids"][arrays["test_idx"]],
        y_true,
        y_pred,
        test_prob,
        class_names,
    ).to_csv(args.out_dir / "probabilities_test.csv", index=False)

    n_features = int(build_feature_matrix(arrays, arrays["train_idx"][:1], args.modality).shape[1])
    summary = {
        "method": f"MLP_{args.modality}",
        "modality": args.modality,
        "balance_by": args.balance_by,
        "domain_reg": args.domain_reg,
        "domain_lambda": args.domain_lambda,
        **{f"test_{key}": value for key, value in test_summary.items()},
        "test_loss": float(test_loss),
        "best_val_macro_f1": train_info["best_val_macro_f1"],
        "best_epoch": train_info["best_epoch"],
        "epochs_ran": int(len(history)),
        "selected_model": train_info["selected_model"],
        "swa_n_checkpoints": train_info["swa_n_checkpoints"],
        "device": train_info["device"],
        "n_train": int(len(arrays["train_idx"])),
        "n_val": int(len(arrays["val_idx"])),
        "n_test": int(len(arrays["test_idx"])),
        "n_genes": int(arrays["gene"].shape[1]),
        "n_proteins": int(arrays["protein"].shape[1]),
        "n_features": n_features,
        "n_classes": int(len(arrays["label_encoder"].classes_)),
        "runtime_seconds": float(time.perf_counter() - start),
    }
    pd.DataFrame([summary]).to_csv(args.out_dir / "results_summary.csv", index=False)
    with open(args.out_dir / "config.json", "w") as handle:
        json.dump(json_ready_args(args), handle, indent=2)

    print(pd.DataFrame([summary]).to_string(index=False))


if __name__ == "__main__":
    main()
