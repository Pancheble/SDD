from __future__ import annotations

import torch
import torch.nn.functional as F


class Center:
    def __init__(self, dim: int, momentum: float = 0.9, device: str | torch.device = "cpu"):
        self.momentum = momentum
        self.value = torch.zeros(dim, device=device)

    @torch.no_grad()
    def update(self, teacher_logits: torch.Tensor) -> None:
        batch_center = teacher_logits.mean(dim=0)
        self.value.mul_(self.momentum).add_(batch_center, alpha=1.0 - self.momentum)

    def __call__(self) -> torch.Tensor:
        return self.value


def dino_loss(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    center: Center | None,
    teacher_temp: float = 0.04,
    student_temp: float = 0.1,
    use_centering: bool = True,
    use_sharpening: bool = True,
) -> torch.Tensor:
    t_logits = teacher_logits
    if use_centering and center is not None:
        t_logits = t_logits - center().detach()
    t_temp = teacher_temp if use_sharpening else student_temp
    s_temp = student_temp
    teacher_probs = F.softmax(t_logits / max(t_temp, 1e-6), dim=-1)
    student_log_probs = F.log_softmax(student_logits / max(s_temp, 1e-6), dim=-1)
    return -(teacher_probs * student_log_probs).sum(dim=-1).mean()
