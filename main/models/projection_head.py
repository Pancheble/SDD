"""
models/projection_head.py
2-layer MLP projection head (논문 Section 3.3.2).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """
    2-layer MLP: in_dim → hidden_dim → out_dim (L2 normalized).
    Student / Teacher 각각 독립적으로 보유.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 2048, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
