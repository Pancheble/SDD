from __future__ import annotations

from torchmetrics.image.fid import FrechetInceptionDistance
import torch


class FIDEvaluator:
    def __init__(self, device="cuda"):
        self.metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    @torch.no_grad()
    def update_real(self, imgs: torch.Tensor):
        self.metric.update(imgs, real=True)

    @torch.no_grad()
    def update_fake(self, imgs: torch.Tensor):
        self.metric.update(imgs, real=False)

    def compute(self) -> float:
        return float(self.metric.compute().item())
