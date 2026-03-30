"""
losses/sdd_loss.py
SDD 전체 목적함수.

L_SDD = L_MSE + λ · w(t) · L_DINO(p_t, p_s)

논문 Section 3.3.5.
"""
import torch
import torch.nn as nn
from .diffusion_loss import DiffusionLoss
from .dino_loss import DINOLoss, timestep_gate


class SDDLoss(nn.Module):
    def __init__(
        self,
        # DINO 파라미터
        out_dim: int = 256,
        tau_student: float = 0.1,
        tau_teacher: float = 0.04,
        center_momentum: float = 0.9,
        centering_enabled: bool = True,
        # 타임스텝 게이트
        gate_enabled: bool = True,
        gate_type: str = "hard",
        t_min_ratio: float = 0.1,
        t_max_ratio: float = 0.6,
        gate_beta: float = 50.0,
        # 손실 균형
        lambda_dino: float = 0.5,
        # 총 타임스텝
        T: int = 1000,
    ):
        super().__init__()
        self.lambda_dino = lambda_dino
        self.gate_enabled = gate_enabled
        self.gate_type = gate_type
        self.t_min_ratio = t_min_ratio
        self.t_max_ratio = t_max_ratio
        self.gate_beta = gate_beta
        self.T = T

        self.diffusion_loss = DiffusionLoss()
        self.dino_loss = DINOLoss(
            out_dim=out_dim,
            tau_student=tau_student,
            tau_teacher=tau_teacher,
            center_momentum=center_momentum,
            centering_enabled=centering_enabled,
        )

    def forward(
        self,
        noise_pred: torch.Tensor,      # [B, C, H, W]
        noise_target: torch.Tensor,    # [B, C, H, W]
        student_proj: torch.Tensor,    # [B, K]
        teacher_proj: torch.Tensor,    # [B, K]
        t: torch.Tensor,               # [B] 타임스텝
    ) -> dict:
        """
        Returns:
            dict with keys:
              'total', 'l_mse', 'l_dino', 'gate_mean'
        """
        # ── Denoising MSE loss ────────────────────────────────────────────
        l_mse = self.diffusion_loss(noise_pred, noise_target)

        # ── DINO loss + 게이트 ────────────────────────────────────────────
        if self.gate_enabled:
            w = timestep_gate(
                t, self.T,
                t_min_ratio=self.t_min_ratio,
                t_max_ratio=self.t_max_ratio,
                gate_type=self.gate_type,
                beta=self.gate_beta,
            )  # [B]
        else:
            w = torch.ones(t.shape[0], device=t.device)

        l_dino_raw = self.dino_loss(student_proj, teacher_proj)
        # 게이트를 배치 평균에 적용
        # (배치 내 일부 샘플만 gate=0일 수 있으므로 mean(w)로 스케일링)
        gate_mean = w.mean()
        l_dino = l_dino_raw * gate_mean

        total = l_mse + self.lambda_dino * l_dino

        return {
            "total": total,
            "l_mse": l_mse.detach(),
            "l_dino": l_dino.detach(),
            "gate_mean": gate_mean.detach(),
        }
