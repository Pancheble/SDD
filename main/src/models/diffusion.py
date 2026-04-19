from __future__ import annotations

import torch
import torch.nn.functional as F


def make_beta_schedule(timesteps: int = 1000, schedule: str = "linear") -> torch.Tensor:
    if schedule == "linear":
        return torch.linspace(1e-4, 0.02, timesteps)
    raise ValueError(f"Unknown beta schedule: {schedule}")


class Diffusion:
    def __init__(self, timesteps: int = 1000, beta_schedule: str = "linear", device: str | torch.device = "cpu"):
        self.timesteps = timesteps
        self.betas = make_beta_schedule(timesteps, beta_schedule).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None):
        if noise is None:
            noise = torch.randn_like(x0)
        s1 = self.sqrt_alphas_cumprod[t][:, None, None, None]
        s2 = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return s1 * x0 + s2 * noise, noise


def diffusion_mse_loss(model, diffusion: Diffusion, x0: torch.Tensor, t: torch.Tensor):
    xt, noise = diffusion.q_sample(x0, t)
    pred = model(xt, t)
    return F.mse_loss(pred, noise)
