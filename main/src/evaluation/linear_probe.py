from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm


class LinearProbe(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


def train_linear_probe(train_feats, train_labels, val_feats, val_labels, num_classes: int, epochs: int = 50, lr: float = 1e-3, device="cuda"):
    model = LinearProbe(train_feats.shape[1], num_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    train_feats = train_feats.to(device)
    train_labels = train_labels.to(device)
    val_feats = val_feats.to(device)
    val_labels = val_labels.to(device)

    for _ in tqdm(range(epochs), desc="linear probe"):
        model.train()
        logits = model(train_feats)
        loss = F.cross_entropy(logits, train_labels)
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        pred = model(val_feats).argmax(dim=-1)
        acc = (pred == val_labels).float().mean().item()
    return model, acc
