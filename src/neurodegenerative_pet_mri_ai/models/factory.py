from __future__ import annotations

from torch import nn

from neurodegenerative_pet_mri_ai.models.fusion import MultimodalFusionModel
from neurodegenerative_pet_mri_ai.models.unimodal import MRIClassifier, PETClassifier


def build_model(model_name: str, embedding_dim: int, num_classes: int) -> nn.Module:
    registry = {
        "mri_only": MRIClassifier,
        "pet_only": PETClassifier,
        "multimodal_fusion": MultimodalFusionModel,
    }
    if model_name not in registry:
        raise ValueError(f"Unknown model_name='{model_name}'. Available: {sorted(registry)}")
    return registry[model_name](embedding_dim=embedding_dim, num_classes=num_classes)
