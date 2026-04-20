from __future__ import annotations

import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def extract_features(
    model,
    loader: DataLoader,
    device,
    feature_layer: str = "bottleneck",
    timestep: int = 0,
):
    """Extract intermediate features from the UNet for an entire dataloader.

    Args:
        model: UNetModel instance
        loader: DataLoader yielding (x, y) batches
        device: target device
        feature_layer: one of "bottleneck", "skip1", "skip2", "decoder1"
        timestep: diffusion timestep at which to extract (default 0 = clean)
    """
    model.eval()
    feats, labels = [], []
    for x, y in loader:
        x = x.to(device)
        t = torch.full((x.size(0),), timestep, device=device, dtype=torch.long)
        _, f = model(x, t, return_features=True, feature_layer=feature_layer)
        feats.append(f.detach().cpu())
        labels.append(y.cpu())
    return torch.cat(feats), torch.cat(labels)
