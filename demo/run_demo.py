#!/usr/bin/env python
"""Run the small preprocessed-input MOSAIC audit demo in a submission package."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PACKAGE = Path(__file__).resolve().parents[1]
MODEL_DIR = PACKAGE / "code/experiments/exp_generalization/original_mosaic_strict"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))
if str(PACKAGE / "code") not in sys.path:
    sys.path.insert(0, str(PACKAGE / "code"))

from run_mosaic_rd_v2 import MosaicRDV2Model  # noqa: E402


DEFAULT_GROUPS = [
    ["CD4 TEM_3", "CD4 TEM_4", "CD4 TCM_3"],
    ["CD8 Naive", "CD8 Naive_2", "CD8 TCM_1"],
]


def load_model(checkpoint_path: Path, config_path: Path, gene_dim: int, protein_dim: int, device: torch.device):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    payload = torch.load(checkpoint_path, map_location=device)
    class_names = [str(value) for value in payload["label_classes"]]
    hsr_config = payload.get("train_info", {}).get("hsr_config") or {
        "sibling_groups": DEFAULT_GROUPS,
        "use_gate": True,
        "gate_floor": float(config.get("hsr_gate_floor", 0.02)),
        "gate_max": float(config.get("hsr_gate_max", 0.20)),
        "use_prob_context": False,
        "candidate_gate_mode": "none",
        "no_hierarchy": False,
    }
    model = MosaicRDV2Model(
        gene_dim=gene_dim,
        protein_dim=protein_dim,
        hidden_dim=int(config.get("hidden_dim", 256)),
        encoder_hidden_dims=[int(value) for value in str(config.get("encoder_hidden_dims", "512")).split(",") if value],
        fusion_hidden_dims=[int(value) for value in str(config.get("fusion_hidden_dims", "512,256")).split(",") if value],
        num_classes=len(class_names),
        dropout=float(config.get("dropout", 0.2)),
        gate_temperature=float(config.get("gate_temperature", 1.0)),
        head_type=str(config.get("head_type", "linear")),
        class_names=class_names,
        hsr_config=hsr_config,
    ).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model, class_names


@torch.no_grad()
def run_demo(model, class_names: list[str], inputs: dict[str, np.ndarray], device: torch.device) -> pd.DataFrame:
    gene = torch.as_tensor(inputs["gene"], dtype=torch.float32, device=device)
    protein = torch.as_tensor(inputs["protein"], dtype=torch.float32, device=device)
    outputs = model(gene, protein)
    final_logits = outputs["final_logits"]
    final_probabilities = torch.softmax(final_logits, dim=1)
    rows = []
    branch_logits = {
        "rna": outputs["rna_logits"],
        "adt": outputs["adt_logits"],
        "fusion": outputs["fusion_logits"],
    }
    for row_index in range(len(gene)):
        branch_predictions = {}
        branch_margins = {}
        for name, logits in branch_logits.items():
            values, indices = torch.topk(logits[row_index], k=2)
            branch_predictions[name] = class_names[int(indices[0])]
            branch_margins[name] = float((values[0] - values[1]).item())
        weights = outputs["weights"][row_index].detach().cpu().numpy()
        gate = outputs.get("hsr_gate")
        delta = outputs.get("hsr_delta_logits")
        rows.append(
            {
                "cell_id": str(inputs["cell_id"][row_index]),
                "final_prediction": class_names[int(final_logits[row_index].argmax())],
                "final_confidence": float(final_probabilities[row_index].max().item()),
                "final_uncertainty": float(1.0 - final_probabilities[row_index].max().item()),
                "rna_prediction": branch_predictions["rna"],
                "adt_prediction": branch_predictions["adt"],
                "fusion_prediction": branch_predictions["fusion"],
                "rna_margin": branch_margins["rna"],
                "adt_margin": branch_margins["adt"],
                "fusion_margin": branch_margins["fusion"],
                "rna_weight": float(weights[0]),
                "adt_weight": float(weights[1]),
                "fusion_weight": float(weights[2]),
                "branch_conflict": len(set(branch_predictions.values())) > 1,
                "hsr_gate": float(gate[row_index, 0].item()) if gate is not None else 0.0,
                "hsr_delta_norm": float(delta[row_index].norm().item()) if delta is not None else 0.0,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, default=PACKAGE)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    package = args.package.resolve()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    inputs = dict(np.load(package / "demo/inputs.npz", allow_pickle=False))
    model, classes = load_model(
        package / "checkpoint/model.pt",
        package / "checkpoint/config.json",
        int(inputs["gene"].shape[1]),
        int(inputs["protein"].shape[1]),
        device,
    )
    output = run_demo(model, classes, inputs, device)
    output_path = package / "demo/audit_record_generated.csv"
    output.to_csv(output_path, index=False)
    print(json.dumps({"device": str(device), "n_cells": len(output), "output": str(output_path)}, indent=2))
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
