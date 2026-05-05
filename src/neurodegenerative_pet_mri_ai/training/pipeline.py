from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW

from neurodegenerative_pet_mri_ai.data.dataset import build_dataloaders
from neurodegenerative_pet_mri_ai.evaluation.metrics import compute_classification_metrics
from neurodegenerative_pet_mri_ai.models.factory import build_model
from neurodegenerative_pet_mri_ai.training.engine import run_epoch
from neurodegenerative_pet_mri_ai.utils.config import load_config
from neurodegenerative_pet_mri_ai.utils.io import ensure_dir, write_json
from neurodegenerative_pet_mri_ai.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multimodal neuroimaging baseline models.")
    parser.add_argument("--config", required=True, help="Path to YAML configuration file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(config["experiment"]["seed"])

    output_dir = Path(config["experiment"]["output_dir"])
    ensure_dir(output_dir)

    dataloaders = build_dataloaders(config)
    model = build_model(
        model_name=config["training"]["model_name"],
        embedding_dim=config["representation"]["embedding_dim"],
        num_classes=config["training"]["num_classes"],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss()

    history: list[dict[str, float]] = []
    for epoch in range(config["training"]["epochs"]):
        train_result = run_epoch(model, dataloaders["train"], criterion, device, optimizer=optimizer)
        val_result = run_epoch(model, dataloaders["val"], criterion, device, optimizer=None)
        val_metrics = compute_classification_metrics(
            val_result.y_true,
            val_result.y_pred,
            average=config["evaluation"]["average"],
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_result.loss,
                "val_loss": val_result.loss,
                "val_accuracy": val_metrics["accuracy"],
                "val_f1": val_metrics["f1"],
            }
        )

    write_json(output_dir / "history.json", {"history": history})
    torch.save(model.state_dict(), output_dir / "model.pt")
