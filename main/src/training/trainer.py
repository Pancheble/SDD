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


def _feat_dim_for_layer(config: Dict[str, Any], layer: str) -> int:
    ch = config["model"]["channels"]
    mults = config["model"]["channel_mults"]
    dim_map = {
        "bottleneck": ch * mults[-1],
        "skip1":      ch * mults[-2] if len(mults) >= 2 else ch * mults[-1],
        "skip2":      ch * mults[-3] if len(mults) >= 3 else ch * mults[-1],
        "decoder1":   ch * mults[-1],
    }
    return dim_map.get(layer, ch * mults[-1])


class SDDTrainer:
    """SDD trainer that works with both single-GPU and Accelerate multi-GPU.

    Usage (single GPU — legacy):
        trainer = SDDTrainer(model, cfg, device="cuda")

    Usage (multi-GPU via Accelerate):
        from accelerate import Accelerator
        accelerator = Accelerator(mixed_precision="fp16")
        trainer = SDDTrainer(model, cfg, accelerator=accelerator)
        # model / optimizer / dataloader are prepared inside train_one_epoch
        # via accelerator.prepare() on first call.
    """

    def __init__(
        self,
        model,
        config: Dict[str, Any],
        device: str | torch.device | None = None,
        accelerator=None,          # accelerate.Accelerator | None
    ):
        self.cfg = config
        self.accelerator = accelerator

        # Resolve device: Accelerator owns the device; fall back to explicit arg.
        if accelerator is not None:
            self.device = accelerator.device
        else:
            self.device = torch.device(device) if device is not None else torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        self.model = model.to(self.device)
        self.diffusion = Diffusion(
            timesteps=config["diffusion"]["timesteps"],
            beta_schedule=config["diffusion"]["beta_schedule"],
            device=self.device,
        )
        self.use_sdd = config["sdd"]["enabled"]
        self.feature_layer: str = config["sdd"].get("feature_layer", "bottleneck")

        # Teacher lives on the same device but is never wrapped by DDP —
        # we always update it via the unwrapped student weights.
        self.teacher = clone_model(self.model) if self.use_sdd else None
        self.proj_student = None
        self.proj_teacher = None
        self.center = None

        if self.use_sdd and config["sdd"].get("use_projection_head", True):
            proj_dim = config["sdd"]["projection_dim"]
            feat_dim = _feat_dim_for_layer(config, self.feature_layer)
            self.proj_student = ProjectionHead(feat_dim, proj_dim=proj_dim).to(self.device)
            self.proj_teacher = ProjectionHead(feat_dim, proj_dim=proj_dim).to(self.device)
            self.proj_teacher.load_state_dict(self.proj_student.state_dict())
            for p in self.proj_teacher.parameters():
                p.requires_grad_(False)
            self.center = Center(
                proj_dim,
                momentum=config["sdd"]["center_momentum"],
                device=self.device,
            )

        self.state = TrainState()
        self._prepared = False   # track whether accelerator.prepare() has run

    # ── helpers ──────────────────────────────────────────────────────────────

    @property
    def is_main(self) -> bool:
        """True on the rank-0 process (or always True when not distributed)."""
        if self.accelerator is not None:
            return self.accelerator.is_main_process
        return True

    def _unwrap(self, module):
        """Strip DDP wrapper to get the raw nn.Module."""
        if self.accelerator is not None:
            return self.accelerator.unwrap_model(module)
        return module

    def _prepare(self, optimizer, train_loader: DataLoader):
        """Run accelerator.prepare() once on first training call."""
        if self._prepared or self.accelerator is None:
            return optimizer, train_loader

        modules = [self.model]
        if self.proj_student is not None:
            modules.append(self.proj_student)

        prepared = self.accelerator.prepare(*modules, optimizer, train_loader)
        # unpack: same number of modules + optimizer + loader
        *prepared_modules, optimizer, train_loader = prepared
        self.model = prepared_modules[0]
        if self.proj_student is not None:
            self.proj_student = prepared_modules[1]

        self._prepared = True
        return optimizer, train_loader

    # ── training ─────────────────────────────────────────────────────────────

    def train_one_epoch(
        self,
        loader: DataLoader,
        optimizer,
        scaler=None,          # ignored when using Accelerate (it handles AMP internally)
        wandb_run=None,
    ):
        optimizer, loader = self._prepare(optimizer, loader)

        self.model.train()
        if self.teacher is not None:
            self.teacher.eval()

        grad_accum = self.cfg["train"]["grad_accum_steps"]
        optimizer.zero_grad(set_to_none=True)
        running = {"loss_total": 0.0, "loss_mse": 0.0, "loss_distill": 0.0, "gate_mean": 0.0}

        # tqdm only on main process to avoid duplicated output
        pbar = tqdm(
            loader,
            desc=f"epoch {self.state.epoch}",
            leave=False,
            disable=not self.is_main,
        )

        use_amp = (
            self.accelerator is None
            and self.cfg["train"]["mixed_precision"]
            and torch.cuda.is_available()
        )

        for i, (x, _) in enumerate(pbar):
            # When using Accelerate the dataloader already places tensors on the
            # correct device; without it we do it manually.
            if self.accelerator is None:
                x = x.to(self.device, non_blocking=True)

            t = torch.randint(0, self.diffusion.timesteps, (x.size(0),), device=self.device)
            xt, noise = self.diffusion.q_sample(x, t)

            # ── forward ──────────────────────────────────────────────────────
            ctx = (
                self.accelerator.autocast()
                if self.accelerator is not None
                else torch.autocast("cuda", dtype=torch.float16, enabled=use_amp)
            )
            with ctx:
                eps_pred, feat_s = self.model(
                    xt, t, return_features=True, feature_layer=self.feature_layer
                )
                student_logits = (
                    self.proj_student(feat_s) if self.proj_student is not None else None
                )

                teacher_logits = None
                if self.teacher is not None:
                    with torch.no_grad():
                        _, feat_t = self.teacher(
                            xt, t, return_features=True, feature_layer=self.feature_layer
                        )
                        teacher_logits = (
                            self.proj_teacher(feat_t)
                            if self.proj_teacher is not None
                            else None
                        )

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
                    soft_mid=self.cfg["sdd"]["gating"].get("soft_mid", 0.4),
                    soft_beta=self.cfg["sdd"]["gating"].get("soft_beta", 0.08),
                )
                loss = loss / grad_accum

            # ── backward ─────────────────────────────────────────────────────
            if self.accelerator is not None:
                self.accelerator.backward(loss)
            elif scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # ── optimizer step ───────────────────────────────────────────────
            if (i + 1) % grad_accum == 0:
                if self.accelerator is not None:
                    optimizer.step()
                elif scaler is not None and scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                # EMA update — always on the unwrapped (non-DDP) student weights
                if self.teacher is not None:
                    student_raw = self._unwrap(self.model)
                    ema_update(
                        self.teacher,
                        student_raw,
                        momentum=self.cfg["sdd"]["teacher_momentum"],
                    )
                    if self.proj_teacher is not None and self.proj_student is not None:
                        proj_s_raw = self._unwrap(self.proj_student)
                        ema_update(
                            self.proj_teacher,
                            proj_s_raw,
                            momentum=self.cfg["sdd"]["teacher_momentum"],
                        )

                    # Center update: gather teacher logits across all GPUs so
                    # the running mean is computed over the full global batch.
                    if self.center is not None and teacher_logits is not None:
                        if self.accelerator is not None and self.accelerator.num_processes > 1:
                            gathered = self.accelerator.gather(teacher_logits.detach())
                        else:
                            gathered = teacher_logits.detach()
                        self.center.update(gathered)

            # ── logging ──────────────────────────────────────────────────────
            for k in running:
                running[k] += metrics[k]

            if self.is_main:
                pbar.set_postfix({k: f"{v / (i + 1):.4f}" for k, v in running.items()})

            log_every = self.cfg["train"]["log_every"]
            if wandb_run is not None and self.is_main and (self.state.step % log_every == 0):
                wandb_run.log({**metrics, "step": self.state.step, "epoch": self.state.epoch})

            self.state.step += 1

        return {k: v / max(len(loader), 1) for k, v in running.items()}

    # ── sampling (inference) ─────────────────────────────────────────────────

    @torch.no_grad()
    def sample(self, n: int, shape: tuple[int, int, int]):
        """DDPM ancestral sampling. Always runs on a single process."""
        model = self._unwrap(self.model)
        model.eval()
        x = torch.randn(n, *shape, device=self.device)
        for t in reversed(range(self.diffusion.timesteps)):
            tt = torch.full((n,), t, device=self.device, dtype=torch.long)
            eps = model(x, tt)
            a  = self.diffusion.alphas[tt][:, None, None, None]
            b  = self.diffusion.betas[tt][:, None, None, None]
            ac = self.diffusion.alphas_cumprod[tt][:, None, None, None]
            x  = (x - (1 - a) / torch.sqrt(1 - ac) * eps) / torch.sqrt(a)
            if t > 0:
                x = x + torch.sqrt(b) * torch.randn_like(x)
        return x.clamp(-1, 1)
