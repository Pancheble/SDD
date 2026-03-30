"""
models/ema.py
EMA (Exponential Moving Average) teacher 관리.
논문 Section 3.3.1: ξ ← m·ξ + (1−m)·θ
"""
import copy
import torch
import torch.nn as nn


class EMAModel:
    """
    EMA wrapper.
    사용법:
        ema = EMAModel(student_model, momentum=0.996)
        # 학습 루프 내:
        ema.update(student_model)
        # 추론 시:
        with ema.average_parameters():
            output = teacher_model(x, t)
    """

    def __init__(self, model: nn.Module, momentum: float = 0.996):
        self.momentum = momentum
        self.shadow = copy.deepcopy(model)
        self.shadow.requires_grad_(False)
        self.shadow.eval()

    @torch.no_grad()
    def update(self, model: nn.Module):
        """학생 모델의 파라미터로 EMA 갱신"""
        for ema_p, model_p in zip(self.shadow.parameters(), model.parameters()):
            ema_p.data.mul_(self.momentum).add_(model_p.data, alpha=1.0 - self.momentum)
        # buffer (BatchNorm 등) 동기화
        for ema_b, model_b in zip(self.shadow.buffers(), model.buffers()):
            ema_b.copy_(model_b)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict):
        self.shadow.load_state_dict(state_dict)

    def __call__(self, *args, **kwargs):
        return self.shadow(*args, **kwargs)
