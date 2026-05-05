from __future__ import annotations

import argparse

import torch
from torch import nn

from neurodegenerative_pet_mri_ai.data.dataset import build_dataloaders
from neurodegenerative_pet_mri_ai.evaluation.metrics import compute_classification_metrics
from neurodegenerative_pet_mri_ai.models.factory import build_model
from neurodegenerative_pet_mri_ai.training.engine import run_epoch
from neurodegenerative_pet_mri_ai.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained neuroimaging model.")
    parser.add_argument("--config", required=True, help="Path to YAML configuration file.")
    parser.add_argument("--checkpoint", required=True, help="Path to a model checkpoint.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    dataloaders = build_dataloaders(config)
    model = build_model(
        model_name=config["training"]["model_name"],
        embedding_dim=config["representation"]["embedding_dim"],
        num_classes=config["training"]["num_classes"],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)

    result = run_epoch(model, dataloaders["test"], nn.CrossEntropyLoss(), device, optimizer=None)
    metrics = compute_classification_metrics(
        result.y_true,
        result.y_pred,
        average=config["evaluation"]["average"],
    )
    print(metrics)
