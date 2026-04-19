from __future__ import annotations

import torch


def timestep_gate(
    t: torch.Tensor,
    timesteps: int,
    mode: str = "hard",
    t_min: float = 0.1,
    t_max: float = 0.6,
    soft_mid: float = 0.4,
    soft_beta: float = 0.08,
) -> torch.Tensor:
    t_norm = t.float() / float(max(timesteps - 1, 1))
    if mode == "hard":
        return ((t_norm >= t_min) & (t_norm <= t_max)).float()
    if mode == "soft":
        left = torch.sigmoid((t_norm - t_min) / max(soft_beta, 1e-6))
        right = torch.sigmoid((t_max - t_norm) / max(soft_beta, 1e-6))
        return left * right
    return torch.ones_like(t_norm)
