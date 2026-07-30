#!/usr/bin/env python
"""MOSAIC-RD v2: donor-invariant prototype late fusion."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset

from run_l3_strict_mosaic import build_strict_arrays, set_seed
from strict_array_cache import build_or_load_strict_arrays, json_ready_args


def parse_hidden_dims(value: str) -> list[int]:
    dims = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not dims:
        raise ValueError("hidden dims must contain at least one integer")
    return dims


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


def make_mlp(input_dim: int, hidden_dims: list[int], output_dim: int, dropout: float, final_activation: bool = True) -> torch.nn.Sequential:
    layers: list[torch.nn.Module] = []
    previous_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.extend([torch.nn.Linear(previous_dim, hidden_dim), torch.nn.BatchNorm1d(hidden_dim), torch.nn.ReLU(), torch.nn.Dropout(dropout)])
        previous_dim = hidden_dim
    layers.append(torch.nn.Linear(previous_dim, output_dim))
    if final_activation:
        layers.extend([torch.nn.BatchNorm1d(output_dim), torch.nn.ReLU()])
    return torch.nn.Sequential(*layers)


def branch_margin(logits: torch.Tensor) -> torch.Tensor:
    top2 = torch.topk(logits, k=2, dim=1).values
    return top2[:, 0:1] - top2[:, 1:2]


def margin_reliability_weights(
    rna_logits: torch.Tensor,
    adt_logits: torch.Tensor,
    fusion_logits: torch.Tensor,
    availability_mask: torch.Tensor,
    branch_bias: torch.Tensor | None = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    margins = torch.cat([branch_margin(rna_logits), branch_margin(adt_logits), branch_margin(fusion_logits)], dim=1)
    if branch_bias is not None:
        margins = margins + branch_bias.view(1, 3)
    branch_mask = torch.stack(
        [
            availability_mask[:, 0],
            availability_mask[:, 1],
            availability_mask[:, 0] * availability_mask[:, 1],
        ],
        dim=1,
    ).to(device=margins.device, dtype=margins.dtype)
    margins = margins / max(float(temperature), 1e-6)
    margins = margins.masked_fill(branch_mask <= 0, -1e9)
    return torch.softmax(margins, dim=1)


class CosinePrototypeClassifier(torch.nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int, init_scale: float = 10.0) -> None:
        super().__init__()
        self.prototypes = torch.nn.Parameter(torch.randn(num_classes, hidden_dim) * 0.02)
        self.logit_scale = torch.nn.Parameter(torch.tensor(float(np.log(init_scale)), dtype=torch.float32))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = F.normalize(features, dim=1)
        prototypes = F.normalize(self.prototypes, dim=1)
        return self.logit_scale.exp().clamp(max=100.0) * features.matmul(prototypes.T)


def make_classifier(head_type: str, hidden_dim: int, num_classes: int) -> torch.nn.Module:
    if head_type == "prototype":
        return CosinePrototypeClassifier(hidden_dim, num_classes)
    if head_type == "linear":
        return torch.nn.Linear(hidden_dim, num_classes)
    raise ValueError(f"Unsupported head_type: {head_type}")


class HierarchyAwareSiblingRefinementHead(torch.nn.Module):
    """Learn local logit refinements inside predeclared sibling class groups."""

    def __init__(
        self,
        hidden_dim: int,
        class_names: list[str],
        sibling_groups: list[list[str]],
        use_gate: bool = True,
        no_hierarchy: bool = False,
        gate_floor: float = 0.0,
        gate_max: float = 1.0,
        use_prob_context: bool = False,
        candidate_gate_mode: str = "none",
        calibration_rules: list[dict] | None = None,
        calibration_epsilon: float = 1e-4,
    ) -> None:
        super().__init__()
        self.class_names = [str(name) for name in class_names]
        self.num_classes = len(self.class_names)
        self.use_gate = use_gate
        self.no_hierarchy = no_hierarchy
        self.gate_floor = float(gate_floor)
        self.gate_max = float(gate_max)
        self.use_prob_context = use_prob_context
        self.candidate_gate_mode = str(candidate_gate_mode)
        self.calibration_rules = list(calibration_rules or [])
        self.calibration_epsilon = float(calibration_epsilon)
        if self.candidate_gate_mode not in {"none", "active_top1", "active_top2"}:
            raise ValueError(f"Unsupported HSR candidate_gate_mode: {candidate_gate_mode}")
        if self.gate_floor < 0 or self.gate_max <= 0 or self.gate_floor > self.gate_max:
            raise ValueError("HSR gate requires 0 <= gate_floor <= gate_max")
        name_to_index = {name: idx for idx, name in enumerate(self.class_names)}
        if no_hierarchy:
            active_indices = list(range(self.num_classes))
        else:
            active_indices = []
            for group in sibling_groups:
                group_indices = []
                for name in group:
                    if name not in name_to_index:
                        raise ValueError(f"Unknown HSR sibling class: {name}")
                    group_indices.append(name_to_index[name])
                active_indices.extend(group_indices)
        active_indices = sorted(set(active_indices))
        if not active_indices:
            raise ValueError("HSR requires at least one active class")
        self.register_buffer("active_indices", torch.tensor(active_indices, dtype=torch.long), persistent=False)
        parsed_rules = []
        for rule in self.calibration_rules:
            source_name = str(rule["source_class"])
            target_name = str(rule["target_class"])
            if source_name not in name_to_index or target_name not in name_to_index:
                raise ValueError(f"Unknown HSR calibration rule classes: {rule}")
            parsed_rules.append(
                (
                    name_to_index[source_name],
                    name_to_index[target_name],
                    float(rule["min_target_prob"]),
                    int(rule["max_target_rank"]),
                )
            )
        self._parsed_calibration_rules = parsed_rules
        delta_input_dim = hidden_dim + self.num_classes if use_prob_context else hidden_dim
        self.delta_head = torch.nn.Linear(delta_input_dim, len(active_indices))
        self.gate_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim + self.num_classes, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
            torch.nn.Sigmoid(),
        )

    def _calibration_forward(self, base_logits: torch.Tensor) -> dict[str, torch.Tensor]:
        sparse_delta = base_logits.new_zeros(base_logits.shape)
        gate = base_logits.new_zeros((base_logits.shape[0], 1))
        if self.training or not self._parsed_calibration_rules:
            return {
                "refined_logits": base_logits,
                "delta_logits": sparse_delta,
                "gate": gate,
            }
        probabilities = torch.softmax(base_logits.detach(), dim=1)
        order = torch.argsort(probabilities, dim=1, descending=True)
        ranks = torch.empty_like(order)
        rank_values = torch.arange(1, self.num_classes + 1, device=base_logits.device).view(1, -1).expand_as(order)
        ranks.scatter_(1, order, rank_values)
        pred = base_logits.detach().argmax(dim=1)
        for source, target, min_target_prob, max_target_rank in self._parsed_calibration_rules:
            target_rank = ranks[:, target]
            mask = (
                (pred == int(source))
                & (probabilities[:, target] >= float(min_target_prob))
                & (target_rank <= int(max_target_rank))
            )
            if not bool(mask.any().item()):
                continue
            required_delta = (base_logits[mask, source] - base_logits[mask, target]).clamp_min(0.0) + self.calibration_epsilon
            sparse_delta[mask, target] = torch.maximum(sparse_delta[mask, target], required_delta)
            gate[mask, 0] = 1.0
            pred[mask] = int(target)
        return {
            "refined_logits": base_logits + sparse_delta,
            "delta_logits": sparse_delta,
            "gate": gate,
        }

    def forward(self, fusion_embed: torch.Tensor, base_logits: torch.Tensor) -> dict[str, torch.Tensor]:
        if self._parsed_calibration_rules:
            return self._calibration_forward(base_logits)
        sparse_delta = base_logits.new_zeros(base_logits.shape)
        base_probabilities = torch.softmax(base_logits.detach(), dim=1)
        if self.use_prob_context:
            delta_input = torch.cat([fusion_embed, base_probabilities], dim=1)
        else:
            delta_input = fusion_embed
        local_delta = self.delta_head(delta_input)
        sparse_delta[:, self.active_indices.to(base_logits.device)] = local_delta
        if self.use_gate:
            gate_input = torch.cat([fusion_embed, base_probabilities], dim=1)
            raw_gate = self.gate_head(gate_input)
            gate = self.gate_floor + (self.gate_max - self.gate_floor) * raw_gate
        else:
            gate = base_logits.new_ones((base_logits.shape[0], 1))
        if self.candidate_gate_mode in {"active_top1", "active_top2"}:
            top_k = 1 if self.candidate_gate_mode == "active_top1" else 2
            top_classes = torch.topk(base_logits.detach(), k=min(top_k, self.num_classes), dim=1).indices
            active = self.active_indices.to(base_logits.device)
            candidate_mask = (top_classes[:, :, None] == active[None, None, :]).any(dim=2).any(dim=1).to(base_logits.dtype).unsqueeze(1)
            gate = gate * candidate_mask
        refined_logits = base_logits + gate * sparse_delta
        return {
            "refined_logits": refined_logits,
            "delta_logits": sparse_delta,
            "gate": gate,
        }


def mmd_domain_loss(features: torch.Tensor, groups: torch.Tensor, bandwidths: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)) -> torch.Tensor:
    unique_groups = torch.unique(groups)
    grouped = [F.normalize(features[groups == group], dim=1) for group in unique_groups if int((groups == group).sum().item()) >= 2]
    if len(grouped) < 2:
        return features.new_zeros(())

    loss = features.new_zeros(())
    n_pairs = 0
    for i in range(len(grouped)):
        for j in range(i + 1, len(grouped)):
            x = grouped[i]
            y = grouped[j]
            xx = torch.cdist(x, x, p=2).pow(2)
            yy = torch.cdist(y, y, p=2).pow(2)
            xy = torch.cdist(x, y, p=2).pow(2)
            pair_loss = features.new_zeros(())
            for bandwidth in bandwidths:
                gamma = 1.0 / (2.0 * bandwidth * bandwidth)
                pair_loss = pair_loss + torch.exp(-gamma * xx).mean() + torch.exp(-gamma * yy).mean() - 2.0 * torch.exp(-gamma * xy).mean()
            loss = loss + pair_loss / len(bandwidths)
            n_pairs += 1
    return loss / max(n_pairs, 1)


class MosaicRDV2Model(torch.nn.Module):
    def __init__(
        self,
        gene_dim: int,
        protein_dim: int,
        hidden_dim: int,
        encoder_hidden_dims: list[int],
        fusion_hidden_dims: list[int],
        num_classes: int,
        dropout: float,
        gate_temperature: float = 1.0,
        head_type: str = "prototype",
        class_names: list[str] | None = None,
        hsr_config: dict | None = None,
    ) -> None:
        super().__init__()
        self.rna_encoder = make_mlp(gene_dim, encoder_hidden_dims, hidden_dim, dropout)
        self.adt_encoder = make_mlp(protein_dim, encoder_hidden_dims, hidden_dim, dropout)
        self.fusion_encoder = make_mlp(hidden_dim * 2, fusion_hidden_dims, hidden_dim, dropout)
        self.rna_classifier = make_classifier(head_type, hidden_dim, num_classes)
        self.adt_classifier = make_classifier(head_type, hidden_dim, num_classes)
        self.fusion_classifier = make_classifier(head_type, hidden_dim, num_classes)
        self.branch_bias = torch.nn.Parameter(torch.zeros(3))
        self.gate_temperature = gate_temperature
        self.head_type = head_type
        self.hsr_head = None
        if hsr_config is not None:
            if class_names is None:
                raise ValueError("class_names are required when hsr_config is provided")
            self.hsr_head = HierarchyAwareSiblingRefinementHead(
                hidden_dim=hidden_dim,
                class_names=list(class_names),
                sibling_groups=hsr_config.get("sibling_groups", []),
                use_gate=bool(hsr_config.get("use_gate", True)),
                no_hierarchy=bool(hsr_config.get("no_hierarchy", False)),
                gate_floor=float(hsr_config.get("gate_floor", 0.0)),
                gate_max=float(hsr_config.get("gate_max", 1.0)),
                use_prob_context=bool(hsr_config.get("use_prob_context", False)),
                candidate_gate_mode=str(hsr_config.get("candidate_gate_mode", "none")),
                calibration_rules=hsr_config.get("calibration_rules"),
                calibration_epsilon=float(hsr_config.get("calibration_epsilon", 1e-4)),
            )

    def forward(self, gene: torch.Tensor, protein: torch.Tensor, availability_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if availability_mask is None:
            availability_mask = torch.ones(gene.shape[0], 2, device=gene.device)
        availability_mask = availability_mask.to(device=gene.device, dtype=gene.dtype)
        gene = gene * availability_mask[:, 0:1]
        protein = protein * availability_mask[:, 1:2]

        rna_embed = self.rna_encoder(gene)
        adt_embed = self.adt_encoder(protein)
        fusion_embed = self.fusion_encoder(torch.cat([rna_embed, adt_embed], dim=1))
        rna_logits = self.rna_classifier(rna_embed)
        adt_logits = self.adt_classifier(adt_embed)
        fusion_logits = self.fusion_classifier(fusion_embed)
        weights = margin_reliability_weights(
            rna_logits,
            adt_logits,
            fusion_logits,
            availability_mask,
            branch_bias=self.branch_bias,
            temperature=self.gate_temperature,
        )
        final_logits = weights[:, 0:1] * rna_logits + weights[:, 1:2] * adt_logits + weights[:, 2:3] * fusion_logits
        outputs = {
            "final_logits": final_logits,
            "base_final_logits": final_logits,
            "rna_logits": rna_logits,
            "adt_logits": adt_logits,
            "fusion_logits": fusion_logits,
            "weights": weights,
            "rna_embed": rna_embed,
            "adt_embed": adt_embed,
            "fusion_embed": fusion_embed,
        }
        if self.hsr_head is not None:
            hsr_outputs = self.hsr_head(fusion_embed, final_logits)
            outputs["final_logits"] = hsr_outputs["refined_logits"]
            outputs["hsr_delta_logits"] = hsr_outputs["delta_logits"]
            outputs["hsr_gate"] = hsr_outputs["gate"]
        return outputs


def apply_modality_dropout(batch_size: int, dropout_prob: float, device: torch.device, seed: int | None = None) -> torch.Tensor:
    if dropout_prob <= 0:
        return torch.ones(batch_size, 2, device=device)
    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(seed)
    choices = torch.rand(batch_size, generator=generator)
    mask = torch.ones(batch_size, 2)
    half = dropout_prob / 2.0
    mask[choices < half, 1] = 0.0
    mask[(choices >= half) & (choices < dropout_prob), 0] = 0.0
    return mask.to(device)


def make_loader(arrays: dict, indices: np.ndarray, batch_size: int, shuffle: bool, seed: int, include_donors: bool = False) -> DataLoader:
    tensors = [
        torch.tensor(arrays["gene"][indices], dtype=torch.float32),
        torch.tensor(arrays["protein"][indices], dtype=torch.float32),
        torch.tensor(arrays["labels"][indices], dtype=torch.long),
        torch.tensor(indices, dtype=torch.long),
    ]
    if include_donors:
        _, donor_groups = np.unique(arrays["donors"][indices], return_inverse=True)
        tensors.append(torch.tensor(donor_groups, dtype=torch.long))
    dataset = TensorDataset(*tensors)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False, generator=generator if shuffle else None)


@torch.no_grad()
def evaluate(model: MosaicRDV2Model, loader: DataLoader, device: torch.device, mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    labels, predictions, weights, local_indices = [], [], [], []
    for gene, protein, y, idx in loader:
        gene = gene.to(device)
        protein = protein.to(device)
        if mode == "full":
            mask = torch.ones(len(y), 2, device=device)
        elif mode == "gene":
            mask = torch.tensor([[1.0, 0.0]], device=device).repeat(len(y), 1)
        elif mode == "protein":
            mask = torch.tensor([[0.0, 1.0]], device=device).repeat(len(y), 1)
        else:
            raise ValueError(f"Unsupported eval mode: {mode}")
        outputs = model(gene, protein, availability_mask=mask)
        labels.append(y.numpy())
        predictions.append(outputs["final_logits"].argmax(dim=1).cpu().numpy())
        weights.append(outputs["weights"].cpu().numpy())
        local_indices.append(idx.numpy())
    return np.concatenate(labels), np.concatenate(predictions), np.concatenate(weights), np.concatenate(local_indices)


def train_model(arrays: dict, args: argparse.Namespace) -> tuple[MosaicRDV2Model, list[dict], dict]:
    if args.domain_loss_weight > 0 and "donors" not in arrays:
        raise ValueError("--domain-loss-weight > 0 requires a cache containing arrays['donors']")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = MosaicRDV2Model(
        gene_dim=arrays["gene"].shape[1],
        protein_dim=arrays["protein"].shape[1],
        hidden_dim=args.hidden_dim,
        encoder_hidden_dims=parse_hidden_dims(args.encoder_hidden_dims),
        fusion_hidden_dims=parse_hidden_dims(args.fusion_hidden_dims),
        num_classes=len(arrays["label_encoder"].classes_),
        dropout=args.dropout,
        gate_temperature=args.gate_temperature,
        head_type=args.head_type,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    train_loader = make_loader(arrays, arrays["train_idx"], args.batch_size, True, args.seed, include_donors=args.domain_loss_weight > 0)
    val_loader = make_loader(arrays, arrays["val_idx"], args.batch_size, False, args.seed)

    best_state = None
    best_val_macro = -1.0
    best_epoch = 0
    stale_epochs = 0
    history = []
    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0
        weight_sum = torch.zeros(3, device=device)
        for batch in train_loader:
            if len(batch) == 5:
                gene, protein, y, _, donor_groups = batch
                donor_groups = donor_groups.to(device)
            else:
                gene, protein, y, _ = batch
                donor_groups = None
            gene = gene.to(device)
            protein = protein.to(device)
            y = y.to(device)
            mask = apply_modality_dropout(len(y), args.modality_dropout, device, seed=args.seed + step)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(gene, protein, availability_mask=mask)
            loss = criterion(outputs["final_logits"], y)
            loss = loss + args.branch_loss_weight * (criterion(outputs["rna_logits"], y) + criterion(outputs["adt_logits"], y))
            loss = loss + args.fusion_loss_weight * criterion(outputs["fusion_logits"], y)
            if args.domain_loss_weight > 0 and donor_groups is not None:
                loss = loss + args.domain_loss_weight * mmd_domain_loss(outputs["fusion_embed"], donor_groups)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += float(loss.item()) * len(y)
            total += len(y)
            correct += int((outputs["final_logits"].argmax(dim=1) == y).sum().item())
            weight_sum += outputs["weights"].detach().sum(dim=0)
            step += 1

        val_true, val_pred, _, _ = evaluate(model, val_loader, device, mode="full")
        val_summary = summarize_predictions(val_true, val_pred)
        mean_weights = (weight_sum / max(total, 1)).detach().cpu().numpy()
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(total, 1),
            "train_accuracy": correct / max(total, 1),
            "val_accuracy": val_summary["accuracy"],
            "val_weighted_f1": val_summary["weighted_f1"],
            "val_macro_f1": val_summary["macro_f1"],
            "train_mean_rna_weight": float(mean_weights[0]),
            "train_mean_adt_weight": float(mean_weights[1]),
            "train_mean_fusion_weight": float(mean_weights[2]),
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} train_loss={row['train_loss']:.4f} "
            f"val_macro_f1={row['val_macro_f1']:.4f} val_weighted_f1={row['val_weighted_f1']:.4f} "
            f"w=({row['train_mean_rna_weight']:.2f},{row['train_mean_adt_weight']:.2f},{row['train_mean_fusion_weight']:.2f})",
            flush=True,
        )
        if val_summary["macro_f1"] > best_val_macro:
            best_val_macro = val_summary["macro_f1"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, {"best_val_macro_f1": float(best_val_macro), "best_epoch": int(best_epoch), "device": str(device)}


def save_weight_summary(out_dir: Path, arrays: dict, y_true: np.ndarray, weights: np.ndarray) -> None:
    labels = arrays["label_encoder"].inverse_transform(y_true)
    frame = pd.DataFrame({"label": labels, "rna_weight": weights[:, 0], "adt_weight": weights[:, 1], "fusion_weight": weights[:, 2]})
    summary = (
        frame.groupby("label", as_index=False)
        .agg(
            mean_rna_weight=("rna_weight", "mean"),
            mean_adt_weight=("adt_weight", "mean"),
            mean_fusion_weight=("fusion_weight", "mean"),
            n_cells=("label", "size"),
        )
        .sort_values("mean_fusion_weight", ascending=False)
    )
    summary.to_csv(out_dir / "weight_summary_by_class.csv", index=False)


def make_prediction_frame(arrays: dict, y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray, local_indices: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cell_id": arrays["cell_ids"][local_indices],
            "label": arrays["label_encoder"].inverse_transform(y_true),
            "prediction": arrays["label_encoder"].inverse_transform(y_pred),
            "rna_weight": weights[:, 0],
            "adt_weight": weights[:, 1],
            "fusion_weight": weights[:, 2],
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gene-path", type=Path, default=Path("data/pbmc/pbmc_gene.h5ad"))
    parser.add_argument("--protein-path", type=Path, default=Path("data/pbmc/pbmc_protein.h5ad"))
    parser.add_argument("--label-column", default="celltype.l3")
    parser.add_argument("--out-dir", type=Path, default=Path("results/exp_generalization/original_mosaic_strict/l3_mosaic_rd_v2"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--n-genes", type=int, default=3000)
    parser.add_argument("--n-quantiles", type=int, default=1000)
    parser.add_argument("--min-cells-per-class", type=int, default=2)
    parser.add_argument("--min-cells-after-subsample", type=int, default=5)
    parser.add_argument("--max-cells", type=int, default=100000)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--encoder-hidden-dims", default="512")
    parser.add_argument("--fusion-hidden-dims", default="512,256")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--modality-dropout", type=float, default=0.15)
    parser.add_argument("--branch-loss-weight", type=float, default=0.35)
    parser.add_argument("--fusion-loss-weight", type=float, default=0.5)
    parser.add_argument("--domain-loss-weight", type=float, default=0.01)
    parser.add_argument("--gate-temperature", type=float, default=1.0)
    parser.add_argument("--head-type", choices=["prototype", "linear"], default="prototype")
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--cache-path", type=Path, default=None)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    start = time.perf_counter()
    random.seed(args.seed)
    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    arrays = build_or_load_strict_arrays(args, build_strict_arrays)
    model, history, train_info = train_model(arrays, args)
    pd.DataFrame(history).to_csv(args.out_dir / "training_history.csv", index=False)

    device = torch.device(train_info["device"])
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "train_info": train_info,
            "label_classes": arrays["label_encoder"].classes_.tolist(),
            "args": json_ready_args(args),
        },
        args.out_dir / "model.pt",
    )

    val_loader = make_loader(arrays, arrays["val_idx"], args.batch_size, False, args.seed)
    val_true, val_pred, val_weights, val_local_indices = evaluate(model.to(device), val_loader, device, mode="full")
    make_prediction_frame(arrays, val_true, val_pred, val_weights, val_local_indices).to_csv(args.out_dir / "predictions_val_full.csv", index=False)

    test_loader = make_loader(arrays, arrays["test_idx"], args.batch_size, False, args.seed)
    rows = []
    full_artifacts = None
    for mode in ["full", "gene", "protein"]:
        y_true, y_pred, weights, local_indices = evaluate(model.to(device), test_loader, device, mode=mode)
        summary = summarize_predictions(y_true, y_pred)
        rows.append(
            {
                "method": "MOSAIC_RD_v2",
                "eval_mode": mode,
                **{f"test_{key}": value for key, value in summary.items()},
                "best_val_macro_f1": train_info["best_val_macro_f1"],
                "best_epoch": train_info["best_epoch"],
                "epochs_ran": int(len(history)),
                "domain_loss_weight": args.domain_loss_weight,
                "gate_temperature": args.gate_temperature,
                "head_type": args.head_type,
                "device": train_info["device"],
                "n_train": int(len(arrays["train_idx"])),
                "n_val": int(len(arrays["val_idx"])),
                "n_test": int(len(arrays["test_idx"])),
                "n_genes": int(arrays["gene"].shape[1]),
                "n_proteins": int(arrays["protein"].shape[1]),
                "n_classes": int(len(arrays["label_encoder"].classes_)),
                "runtime_seconds": float(time.perf_counter() - start),
            }
        )
        if mode == "full":
            full_artifacts = (y_true, y_pred, weights, local_indices)

    pd.DataFrame(rows).to_csv(args.out_dir / "results_summary.csv", index=False)
    if full_artifacts is not None:
        y_true, y_pred, weights, local_indices = full_artifacts
        make_prediction_frame(arrays, y_true, y_pred, weights, local_indices).to_csv(args.out_dir / "predictions_full.csv", index=False)
        save_weight_summary(args.out_dir, arrays, y_true, weights)

    with open(args.out_dir / "config.json", "w") as handle:
        json.dump(json_ready_args(args), handle, indent=2)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
