from __future__ import annotations

import torch.nn as nn


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int = 256, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or max(in_dim, proj_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, x):
        return self.net(x)
