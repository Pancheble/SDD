from __future__ import annotations

import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def extract_features(model, loader: DataLoader, device):
    model.eval()
    feats = []
    labels = []
    for x, y in loader:
        x = x.to(device)
        t = torch.zeros(x.size(0), device=device, dtype=torch.long)
        _, f = model(x, t, return_features=True)
        feats.append(f.detach().cpu())
        labels.append(y.cpu())
    return torch.cat(feats), torch.cat(labels)
