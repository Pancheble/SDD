from __future__ import annotations

from copy import deepcopy
import torch
from torch import nn


def clone_model(model: nn.Module) -> nn.Module:
    teacher = deepcopy(model)
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()
    return teacher


@torch.no_grad()
def ema_update(teacher: nn.Module, student: nn.Module, momentum: float = 0.996) -> None:
    teacher_params = dict(teacher.named_parameters())
    student_params = dict(student.named_parameters())
    for name, t_param in teacher_params.items():
        if name in student_params:
            s_param = student_params[name]
            t_param.data.mul_(momentum).add_(s_param.data, alpha=1.0 - momentum)

    teacher_buffers = dict(teacher.named_buffers())
    student_buffers = dict(student.named_buffers())
    for name, t_buf in teacher_buffers.items():
        if name in student_buffers and t_buf.dtype.is_floating_point:
            t_buf.data.copy_(student_buffers[name].data)
