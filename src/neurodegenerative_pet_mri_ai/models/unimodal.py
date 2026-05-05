from __future__ import annotations

import torch
from torch import nn

from neurodegenerative_pet_mri_ai.models.encoders import DenseNet3DEncoder


class _UnimodalClassifier(nn.Module):
    def __init__(self, embedding_dim: int = 128, num_classes: int = 3) -> None:
        super().__init__()
        self.encoder = DenseNet3DEncoder(embedding_dim=embedding_dim)
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(embedding_dim, num_classes),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        return self.head(features)


class MRIClassifier(_UnimodalClassifier):
    pass


class PETClassifier(_UnimodalClassifier):
    pass
