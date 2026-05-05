from __future__ import annotations

import torch
from torch import nn

from neurodegenerative_pet_mri_ai.models.encoders import DenseNet3DEncoder


class MultimodalFusionModel(nn.Module):
    def __init__(self, embedding_dim: int = 128, num_classes: int = 3) -> None:
        super().__init__()
        self.mri_encoder = DenseNet3DEncoder(embedding_dim=embedding_dim)
        self.pet_encoder = DenseNet3DEncoder(embedding_dim=embedding_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(embedding_dim, num_classes),
        )

    def forward_features(self, mri: torch.Tensor, pet: torch.Tensor) -> torch.Tensor:
        mri_features = self.mri_encoder(mri)
        pet_features = self.pet_encoder(pet)
        return torch.cat([mri_features, pet_features], dim=1)

    def forward(self, mri: torch.Tensor, pet: torch.Tensor) -> torch.Tensor:
        fused = self.forward_features(mri, pet)
        return self.classifier(fused)
