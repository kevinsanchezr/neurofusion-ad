from __future__ import annotations

import torch
from monai.networks.nets import DenseNet121
from torch import nn


class DenseNet3DEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, embedding_dim: int = 128) -> None:
        super().__init__()
        self.backbone = DenseNet121(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=embedding_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
