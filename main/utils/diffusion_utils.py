"""
utils/diffusion_utils.py
Noise schedule, forward process, DDPM/DDIM sampling 유틸리티.
"""
import math
import numpy as np
import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Noise Schedule
# ─────────────────────────────────────────────────────────────────────────────

def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02):
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps: int, s: float = 0.008):
    """
    Improved DDPM cosine schedule (Nichol & Dhariwal 2021).
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)


class DiffusionSchedule:
    """
    사전 계산된 diffusion schedule 버퍼를 관리.

    사용:
        schedule = DiffusionSchedule(timesteps=1000, schedule='cosine')
        x_t, eps = schedule.q_sample(x0, t)
        x0_pred  = schedule.predict_x0(x_t, t, eps_pred)
    """

    def __init__(self, timesteps: int = 1000, schedule: str = "cosine",
                 beta_start: float = 1e-4, beta_end: float = 0.02):
        self.T = timesteps

        if schedule == "linear":
            betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        elif schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # forward process
        self.betas                = betas
        self.alphas               = alphas
        self.alphas_cumprod       = alphas_cumprod
        self.alphas_cumprod_prev  = alphas_cumprod_prev
        self.sqrt_alphas_cumprod       = alphas_cumprod.sqrt()
        self.sqrt_one_minus_alphas_cumprod = (1 - alphas_cumprod).sqrt()

        # posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        self.posterior_mean_coef1 = (
            betas * alphas_cumprod_prev.sqrt() / (1.0 - alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev) * alphas.sqrt() / (1.0 - alphas_cumprod)
        )

    def to(self, device):
        for attr in vars(self):
            val = getattr(self, attr)
            if isinstance(val, torch.Tensor):
                setattr(self, attr, val.to(device))
        return self

    def _extract(self, a: torch.Tensor, t: torch.Tensor, shape):
        """t 인덱스로 배치 추출 후 shape에 맞게 reshape"""
        out = a.gather(0, t)
        return out.reshape(t.shape[0], *((1,) * (len(shape) - 1)))

    # ── Forward process ───────────────────────────────────────────────────

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        """
        x_t = sqrt(ᾱ_t)·x_0 + sqrt(1−ᾱ_t)·ε
        Returns: (x_t, noise)
        """
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        x_t = sqrt_alpha * x0 + sqrt_one_minus * noise
        return x_t, noise

    # ── x_0 prediction ───────────────────────────────────────────────────

    def predict_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor):
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return (x_t - sqrt_one_minus * eps) / sqrt_alpha

    # ── DDPM sampling (reverse process) ──────────────────────────────────

    @torch.no_grad()
    def p_sample(self, model, x_t: torch.Tensor, t: torch.Tensor):
        """단일 DDPM denoising step"""
        betas_t       = self._extract(self.betas, t, x_t.shape)
        sqrt_recip_a  = self._extract((1.0 / self.alphas).sqrt(), t, x_t.shape)
        sqrt_1m_ac    = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)

        eps_pred = model(x_t, t, return_features=False)
        model_mean = sqrt_recip_a * (x_t - betas_t / sqrt_1m_ac * eps_pred)

        if (t == 0).all():
            return model_mean

        posterior_var = self._extract(self.posterior_variance, t, x_t.shape)
        noise = torch.randn_like(x_t)
        return model_mean + posterior_var.sqrt() * noise

    @torch.no_grad()
    def p_sample_loop(self, model, shape, device, progress: bool = True):
        """DDPM 전체 샘플링 루프"""
        from tqdm import tqdm
        x = torch.randn(shape, device=device)
        steps = reversed(range(self.T))
        if progress:
            steps = tqdm(steps, total=self.T, desc="Sampling")
        for i in steps:
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t)
        return x

    # ── DDIM sampling (faster) ────────────────────────────────────────────

    @torch.no_grad()
    def ddim_sample(
        self, model, shape, device,
        ddim_steps: int = 50, eta: float = 0.0, progress: bool = True
    ):
        """DDIM 가속 샘플링"""
        from tqdm import tqdm
        seq = torch.linspace(0, self.T - 1, ddim_steps, dtype=torch.long)
        seq_prev = torch.cat([torch.tensor([-1]), seq[:-1]])

        x = torch.randn(shape, device=device)
        steps = zip(reversed(seq.tolist()), reversed(seq_prev.tolist()))
        if progress:
            steps = tqdm(list(steps), desc="DDIM Sampling")

        for t_val, t_prev_val in steps:
            t     = torch.full((shape[0],), t_val,     device=device, dtype=torch.long)
            t_prev = torch.full((shape[0],), t_prev_val, device=device, dtype=torch.long)

            alpha     = self.alphas_cumprod[t_val]
            alpha_prev = self.alphas_cumprod[t_prev_val] if t_prev_val >= 0 else torch.tensor(1.0)

            eps = model(x, t, return_features=False)
            x0_pred = (x - (1 - alpha).sqrt() * eps) / alpha.sqrt()
            x0_pred = x0_pred.clamp(-1, 1)

            sigma = eta * ((1 - alpha_prev) / (1 - alpha) * (1 - alpha / alpha_prev)).sqrt()
            direction = (1 - alpha_prev - sigma ** 2).sqrt() * eps
            noise = sigma * torch.randn_like(x) if eta > 0 else 0

            x = alpha_prev.sqrt() * x0_pred + direction + noise

        return x
