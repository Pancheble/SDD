"""
losses/dino_loss.py
DINO-style self-distillation loss with centering and sharpening.

논문 Section 3.3.3:
  p_t = softmax((z_t - c) / τ_t)   ← centering + sharpening
  p_s = softmax(z_s / τ_s)
  L_DINO = -Σ p_t · log(p_s)

논문 Section 3.3.4:
  Timestep-adaptive gate: w(t) = 1[t_min ≤ t ≤ t_max] (hard)
                          w(t) = σ(-(t - t_mid) / β)  (soft)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOLoss(nn.Module):
    """
    Args:
        out_dim:            투영 헤드 출력 차원 K
        tau_student:        학생 온도 τ_s (default 0.1)
        tau_teacher:        교사 온도 τ_t (default 0.04, sharpening)
        center_momentum:    센터링 EMA 모멘텀 λ (default 0.9)
        centering_enabled:  False이면 센터링 비활성화 (ablation용)
    """

    def __init__(
        self,
        out_dim: int = 256,
        tau_student: float = 0.1,
        tau_teacher: float = 0.04,
        center_momentum: float = 0.9,
        centering_enabled: bool = True,
    ):
        super().__init__()
        self.tau_s = tau_student
        self.tau_t = tau_teacher
        self.center_momentum = center_momentum
        self.centering_enabled = centering_enabled

        # 센터 벡터 c (running EMA of teacher outputs)
        self.register_buffer("center", torch.zeros(out_dim))

    @torch.no_grad()
    def update_center(self, teacher_output: torch.Tensor):
        """
        배치 단위 EMA 센터 갱신.
        c ← λ·c + (1−λ)·E_batch[z_t]
        """
        batch_center = teacher_output.mean(dim=0)
        self.center = (
            self.center_momentum * self.center
            + (1.0 - self.center_momentum) * batch_center
        )

    def forward(
        self,
        student_proj: torch.Tensor,   # z_s  [B, K]
        teacher_proj: torch.Tensor,   # z_t  [B, K]
    ) -> torch.Tensor:
        """
        Returns:
            L_DINO scalar
        """
        # ── 센터링 (teacher) ──────────────────────────────────────────────
        if self.centering_enabled:
            teacher_centered = teacher_proj - self.center.detach()
        else:
            teacher_centered = teacher_proj

        # ── 분포 생성 ─────────────────────────────────────────────────────
        p_t = F.softmax(teacher_centered / self.tau_t, dim=-1)  # sharpened
        p_s = F.softmax(student_proj / self.tau_s, dim=-1)      # student

        # ── Cross-entropy loss ────────────────────────────────────────────
        # L = -Σ p_t · log(p_s)
        loss = -(p_t * torch.log(p_s + 1e-8)).sum(dim=-1).mean()

        # 센터 갱신 (teacher output으로, centered 이전 값 사용)
        self.update_center(teacher_proj.detach())

        return loss


# ─────────────────────────────────────────────────────────────────────────────
# 타임스텝 게이트
# ─────────────────────────────────────────────────────────────────────────────

def timestep_gate(
    t: torch.Tensor,
    T: int,
    t_min_ratio: float = 0.1,
    t_max_ratio: float = 0.6,
    gate_type: str = "hard",
    beta: float = 50.0,
) -> torch.Tensor:
    """
    논문 Section 3.3.4: 타임스텝 적응형 게이트 w(t).

    Args:
        t:            타임스텝 [B]  (정수, 0~T-1)
        T:            총 타임스텝 수
        t_min_ratio:  하한 비율 (default 0.1)
        t_max_ratio:  상한 비율 (default 0.6)
        gate_type:    'hard' | 'soft'
        beta:         soft gate 선명도

    Returns:
        w: [B] float, 각 샘플에 대한 게이트 값 (0 또는 1 or 연속)
    """
    t_float = t.float() / T   # [0, 1] 정규화

    if gate_type == "hard":
        w = ((t_float >= t_min_ratio) & (t_float <= t_max_ratio)).float()
    elif gate_type == "soft":
        t_mid = (t_min_ratio + t_max_ratio) / 2.0
        w = torch.sigmoid(-beta * (t_float - t_mid))
    else:
        raise ValueError(f"Unknown gate_type: {gate_type}")

    return w
