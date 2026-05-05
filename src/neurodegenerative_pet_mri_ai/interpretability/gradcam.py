from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class GradCAMResult:
    heatmap: torch.Tensor
    target_class: int


class GradCAM3D:
    """Minimal Grad-CAM skeleton for 3D CNNs."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor):
            self.activations = output.detach()

        def backward_hook(
            _module: nn.Module,
            _grad_input: tuple[torch.Tensor, ...],
            grad_output: tuple[torch.Tensor, ...],
        ):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, logits: torch.Tensor, target_class: int) -> GradCAMResult:
        self.model.zero_grad(set_to_none=True)
        score = logits[:, target_class].sum()
        score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations or gradients.")

        weights = self.gradients.mean(dim=(2, 3, 4), keepdim=True)
        heatmap = (weights * self.activations).sum(dim=1, keepdim=True).relu()
        heatmap = heatmap / (heatmap.max() + 1e-8)
        return GradCAMResult(heatmap=heatmap, target_class=target_class)
