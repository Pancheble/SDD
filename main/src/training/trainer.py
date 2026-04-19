from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.models.diffusion import Diffusion
from src.sdd.ema import clone_model, ema_update
from src.sdd.distillation import Center
from src.sdd.projection_head import ProjectionHead
from src.training.losses import total_loss


@dataclass
class TrainState:
    step: int = 0
    epoch: int = 0


class SDDTrainer:
    def __init__(self, model, config: Dict[str, Any], device: str | torch.device):
        self.model = model.to(device)
        self.cfg = config
        self.device = device
        self.diffusion = Diffusion(
            timesteps=config["diffusion"]["timesteps"],
            beta_schedule=config["diffusion"]["beta_schedule"],
            device=device,
        )
        self.use_sdd = config["sdd"]["enabled"]
        self.teacher = clone_model(self.model) if self.use_sdd else None
        self.proj_student = None
        self.proj_teacher = None
        self.center = None

        if self.use_sdd and config["sdd"].get("use_projection_head", True):
            proj_dim = config["sdd"]["projection_dim"]
            feat_dim = config["model"]["channels"] * config["model"]["channel_mults"][-1]
            self.proj_student = ProjectionHead(feat_dim, proj_dim=proj_dim).to(device)
            self.proj_teacher = ProjectionHead(feat_dim, proj_dim=proj_dim).to(device)
            self.proj_teacher.load_state_dict(self.proj_student.state_dict())
            for p in self.proj_teacher.parameters():
                p.requires_grad_(False)
            self.center = Center(proj_dim, momentum=config["sdd"]["center_momentum"], device=device)

        self.state = TrainState()

    def train_one_epoch(self, loader: DataLoader, optimizer, scaler=None, wandb_run=None):
        self.model.train()
        if self.teacher is not None:
            self.teacher.eval()

        pbar = tqdm(loader, desc=f"epoch {self.state.epoch}", leave=False)
        grad_accum = self.cfg["train"]["grad_accum_steps"]
        optimizer.zero_grad(set_to_none=True)

        running = {"loss_total": 0.0, "loss_mse": 0.0, "loss_distill": 0.0, "gate_mean": 0.0}

        for i, (x, _) in enumerate(pbar):
            x = x.to(self.device, non_blocking=True)
            t = torch.randint(0, self.diffusion.timesteps, (x.size(0),), device=self.device)
            xt, noise = self.diffusion.q_sample(x, t)

            use_amp = self.cfg["train"]["mixed_precision"] and torch.cuda.is_available()
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                eps_pred, feat_s = self.model(xt, t, return_features=True)
                student_logits = self.proj_student(feat_s) if self.proj_student is not None else None

                teacher_logits = None
                if self.teacher is not None:
                    with torch.no_grad():
                        _, feat_t = self.teacher(xt, t, return_features=True)
                        teacher_logits = self.proj_teacher(feat_t) if self.proj_teacher is not None else None

                loss, metrics = total_loss(
                    eps_pred=eps_pred,
                    noise=noise,
                    student_logits=student_logits,
                    teacher_logits=teacher_logits,
                    center=self.center,
                    timesteps=t,
                    diffusion_steps=self.diffusion.timesteps,
                    lambda_distill=self.cfg["sdd"]["lambda_distill"],
                    teacher_temp=self.cfg["sdd"]["teacher_temp"],
                    student_temp=self.cfg["sdd"]["student_temp"],
                    use_centering=self.cfg["sdd"]["use_centering"],
                    use_sharpening=self.cfg["sdd"]["use_sharpening"],
                    use_gating=self.cfg["sdd"]["use_gating"],
                    gating_mode=self.cfg["sdd"]["gating"]["mode"],
                    t_min=self.cfg["sdd"]["gating"]["t_min"],
                    t_max=self.cfg["sdd"]["gating"]["t_max"],
                    soft_mid=self.cfg["sdd"]["gating"]["soft_mid"],
                    soft_beta=self.cfg["sdd"]["gating"]["soft_beta"],
                )
                loss = loss / grad_accum

            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (i + 1) % grad_accum == 0:
                if scaler is not None and scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if self.teacher is not None:
                    ema_update(self.teacher, self.model, momentum=self.cfg["sdd"]["teacher_momentum"])
                    if self.proj_teacher is not None and self.proj_student is not None:
                        ema_update(self.proj_teacher, self.proj_student, momentum=self.cfg["sdd"]["teacher_momentum"])
                    if self.center is not None and teacher_logits is not None:
                        self.center.update(teacher_logits.detach())

            for k in running:
                running[k] += metrics[k]

            pbar.set_postfix({k: f"{v / (i + 1):.4f}" for k, v in running.items()})

            if wandb_run is not None and (self.state.step % self.cfg["train"]["log_every"] == 0):
                wandb_run.log({**metrics, "step": self.state.step, "epoch": self.state.epoch})

            self.state.step += 1

        return {k: v / max(len(loader), 1) for k, v in running.items()}

    @torch.no_grad()
    def sample(self, n: int, shape: tuple[int, int, int]):
        self.model.eval()
        x = torch.randn(n, *shape, device=self.device)
        for t in reversed(range(self.diffusion.timesteps)):
            tt = torch.full((n,), t, device=self.device, dtype=torch.long)
            eps = self.model(x, tt)
            a = self.diffusion.alphas[tt][:, None, None, None]
            b = self.diffusion.betas[tt][:, None, None, None]
            ac = self.diffusion.alphas_cumprod[tt][:, None, None, None]
            x = (x - (1 - a) / torch.sqrt(1 - ac) * eps) / torch.sqrt(a)
            if t > 0:
                x = x + torch.sqrt(b) * torch.randn_like(x)
        return x.clamp(-1, 1)
