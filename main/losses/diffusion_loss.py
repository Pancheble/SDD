"""
losses/diffusion_loss.py
표준 diffusion denoising MSE loss.
L_MSE = E[ ||ε − ε_θ(x_t, t)||² ]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        noise_pred: torch.Tensor,
        noise_target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            noise_pred:   ε_θ(x_t, t)  [B, C, H, W]
            noise_target: ε             [B, C, H, W]
        Returns:
            scalar loss
        """
        return F.mse_loss(noise_pred, noise_target)
