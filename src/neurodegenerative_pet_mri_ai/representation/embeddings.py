from __future__ import annotations

from typing import Any

import torch
from torch import nn


@torch.no_grad()
def extract_embeddings(
    model: nn.Module, batch: dict[str, Any], device: torch.device
) -> torch.Tensor:
    model.eval()
    if hasattr(model, "forward_features"):
        if "mri" in batch and "pet" in batch and hasattr(model, "mri_encoder"):
            return model.forward_features(batch["mri"].to(device), batch["pet"].to(device))
        if "pet" in batch and model.__class__.__name__.lower().startswith("pet"):
            return model.forward_features(batch["pet"].to(device))
        return model.forward_features(batch["mri"].to(device))
    raise AttributeError("Model does not expose forward_features for embedding extraction.")
