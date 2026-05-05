from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW

from neurodegenerative_pet_mri_ai.data.dataset import (
    build_dataloaders,
    build_split_manifest_payload,
    create_data_splits,
)
from neurodegenerative_pet_mri_ai.evaluation.metrics import compute_classification_metrics
from neurodegenerative_pet_mri_ai.models.factory import build_model
from neurodegenerative_pet_mri_ai.training.engine import run_epoch
from neurodegenerative_pet_mri_ai.utils.config import load_config
from neurodegenerative_pet_mri_ai.utils.io import ensure_dir, write_json
from neurodegenerative_pet_mri_ai.utils.dataset_paths import resolve_dataset_root
from neurodegenerative_pet_mri_ai.utils.seed import set_seed


LABEL_NAMES = ["Control", "MCI", "AD"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multimodal neuroimaging baseline models.")
    parser.add_argument("--config", required=True, help="Path to YAML configuration file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(config["experiment"]["seed"])
    config["data"]["dataset_root"] = str(resolve_dataset_root(config))

    output_dir = Path(config["experiment"]["output_dir"])
    ensure_dir(output_dir)
    splits = create_data_splits(config)
    write_json(output_dir / "subject_splits.json", build_split_manifest_payload(splits))

    dataloaders = build_dataloaders(config)
    model = build_model(
        model_name=config["training"]["model_name"],
        embedding_dim=config["representation"]["embedding_dim"],
        num_classes=config["training"]["num_classes"],
    )
    device_name = config["training"].get("device", "cpu")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    model.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss(
        weight=_build_class_weights(splits["train"], config["training"]["num_classes"], device)
        if config["training"].get("use_class_weights", True)
        else None
    )

    history: list[dict[str, float | int]] = []
    best_val_balanced_accuracy = float("-inf")
    for epoch in range(config["training"]["epochs"]):
        train_result = run_epoch(model, dataloaders["train"], criterion, device, optimizer=optimizer)
        val_result = run_epoch(model, dataloaders["val"], criterion, device, optimizer=None)
        train_metrics = compute_classification_metrics(
            train_result.y_true,
            train_result.y_pred,
            average=config["evaluation"]["average"],
            label_names=LABEL_NAMES,
        )
        val_metrics = compute_classification_metrics(
            val_result.y_true,
            val_result.y_pred,
            average=config["evaluation"]["average"],
            label_names=LABEL_NAMES,
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_result.loss,
                "val_loss": val_result.loss,
                "train_balanced_accuracy": float(train_metrics["balanced_accuracy"]),
                "train_macro_f1": float(train_metrics["macro_f1"]),
                "val_balanced_accuracy": float(val_metrics["balanced_accuracy"]),
                "val_macro_f1": float(val_metrics["macro_f1"]),
            }
        )

        if float(val_metrics["balanced_accuracy"]) >= best_val_balanced_accuracy:
            best_val_balanced_accuracy = float(val_metrics["balanced_accuracy"])
            torch.save(model.state_dict(), output_dir / "best_model.pt")

    torch.save(model.state_dict(), output_dir / "last_model.pt")
    if (output_dir / "best_model.pt").exists():
        model.load_state_dict(torch.load(output_dir / "best_model.pt", map_location=device))

    test_result = run_epoch(model, dataloaders["test"], criterion, device, optimizer=None)
    test_metrics = compute_classification_metrics(
        test_result.y_true,
        test_result.y_pred,
        average=config["evaluation"]["average"],
        label_names=LABEL_NAMES,
    )

    write_json(output_dir / "history.json", {"history": history})
    write_json(
        output_dir / "metrics.json",
        {
            "train_history": history,
            "test_metrics": test_metrics,
            "device": str(device),
            "label_names": LABEL_NAMES,
        },
    )


def _build_class_weights(
    train_samples: list[dict[str, object]], num_classes: int, device: torch.device
) -> torch.Tensor:
    counts = torch.ones(num_classes, dtype=torch.float32)
    for sample in train_samples:
        counts[int(sample["label"])] += 1.0
    weights = counts.sum() / (counts * float(num_classes))
    return weights.to(device)
