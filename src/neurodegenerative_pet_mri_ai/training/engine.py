from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass
class EpochResult:
    loss: float
    y_true: list[int]
    y_pred: list[int]


def run_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> EpochResult:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []

    for batch in dataloader:
        labels = batch["label"].to(device)
        outputs = _forward_model(model, batch, device)
        loss = criterion(outputs, labels)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        predictions = outputs.argmax(dim=1)
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(predictions.detach().cpu().tolist())

    num_batches = max(len(dataloader), 1)
    return EpochResult(loss=total_loss / num_batches, y_true=y_true, y_pred=y_pred)


def _forward_model(model: nn.Module, batch: dict[str, Any], device: torch.device) -> torch.Tensor:
    if hasattr(model, "mri_encoder") and hasattr(model, "pet_encoder"):
        return model(batch["mri"].to(device), batch["pet"].to(device))

    if model.__class__.__name__.lower().startswith("pet"):
        return model(batch["pet"].to(device))

    return model(batch["mri"].to(device))
