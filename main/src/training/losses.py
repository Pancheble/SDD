from __future__ import annotations

import torch
import torch.nn.functional as F

from src.sdd.distillation import Center, dino_loss
from src.sdd.gating import timestep_gate


def total_loss(
    eps_pred: torch.Tensor,
    noise: torch.Tensor,
    student_logits: torch.Tensor | None,
    teacher_logits: torch.Tensor | None,
    center: Center | None,
    timesteps: torch.Tensor,
    diffusion_steps: int,
    lambda_distill: float = 0.5,
    teacher_temp: float = 0.04,
    student_temp: float = 0.1,
    use_centering: bool = True,
    use_sharpening: bool = True,
    use_gating: bool = True,
    gating_mode: str = "hard",
    t_min: float = 0.1,
    t_max: float = 0.6,
    soft_mid: float = 0.4,
    soft_beta: float = 0.08,
) -> tuple[torch.Tensor, dict]:
    mse = F.mse_loss(eps_pred, noise)
    distill = torch.tensor(0.0, device=eps_pred.device)
    gate = torch.ones_like(timesteps, dtype=torch.float32, device=eps_pred.device)

    if student_logits is not None and teacher_logits is not None and lambda_distill > 0:
        gate = timestep_gate(
            timesteps,
            diffusion_steps,
            mode=gating_mode,
            t_min=t_min,
            t_max=t_max,
            soft_mid=soft_mid,
            soft_beta=soft_beta,
        ).to(eps_pred.device)
        if not use_gating:
            gate = torch.ones_like(gate)

        distill = dino_loss(
            teacher_logits=teacher_logits,
            student_logits=student_logits,
            center=center,
            teacher_temp=teacher_temp,
            student_temp=student_temp,
            use_centering=use_centering,
            use_sharpening=use_sharpening,
        )
        distill = distill * gate.mean()

    total = mse + lambda_distill * distill
    metrics = {
        "loss_total": float(total.detach().cpu()),
        "loss_mse": float(mse.detach().cpu()),
        "loss_distill": float(distill.detach().cpu()),
        "gate_mean": float(gate.mean().detach().cpu()),
    }
    return total, metrics
