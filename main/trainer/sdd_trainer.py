"""
trainer/sdd_trainer.py
SDD 학습 루프 — 논문 Section 3.3.6 pseudocode의 완전한 구현.

핵심 흐름:
  1. x0 샘플링 → 두 뷰 생성
  2. t 샘플링 → x_t 생성
  3. Student forward → (ε_pred, z_s) → proj_s → p_s
  4. Teacher forward (no_grad) → (_, z_t) → proj_t → center → p_t
  5. SDDLoss(ε_pred, ε, p_s, p_t, t) → backward → step
  6. EMA update (teacher, proj_teacher)
"""
import os
import copy
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from omegaconf import DictConfig

from models import UNet, DiT, ProjectionHead, EMAModel
from losses import SDDLoss
from utils import (
    DiffusionSchedule,
    save_checkpoint, load_checkpoint, get_latest_checkpoint, cleanup_old_checkpoints,
    Logger,
)
from data import build_dataloaders


def build_model(cfg: DictConfig, device: torch.device):
    arch = cfg.model.arch.lower()
    if arch == "unet":
        c = cfg.model.unet
        student = UNet(
            in_channels=c.in_channels,
            model_channels=c.model_channels,
            out_channels=c.out_channels,
            num_res_blocks=c.num_res_blocks,
            attention_resolutions=list(c.attention_resolutions),
            dropout=c.dropout,
            channel_mult=tuple(c.channel_mult),
            num_heads=c.num_heads,
        ).to(device)
        feat_dim = c.model_channels * max(c.channel_mult)
    elif arch == "dit":
        c = cfg.model.dit
        student = DiT(
            input_size=c.input_size,
            patch_size=c.patch_size,
            in_channels=c.in_channels,
            hidden_size=c.hidden_size,
            depth=c.depth,
            num_heads=c.num_heads,
            feature_layers=list(c.feature_layers),
        ).to(device)
        feat_dim = c.hidden_size
    else:
        raise ValueError(f"Unknown arch: {arch}")
    return student, feat_dim


def build_lr_scheduler(optimizer, cfg: DictConfig):
    warmup = cfg.training.lr_warmup_steps
    total  = cfg.training.total_steps

    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(3.14159 * progress)).item()))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class SDDTrainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── 출력 디렉토리 ──────────────────────────────────────────────────
        self.output_dir = os.path.join(cfg.output_dir, cfg.experiment_name)
        os.makedirs(self.output_dir, exist_ok=True)

        # ── 데이터 ────────────────────────────────────────────────────────
        self.train_loader, self.val_loader = build_dataloaders(cfg)
        self.train_iter = iter(self.train_loader)

        # ── 모델 빌드 ──────────────────────────────────────────────────────
        self.student, feat_dim = build_model(cfg, self.device)

        # EMA teacher (student의 deepcopy 후 EMA 관리)
        self.ema_teacher = EMAModel(self.student, momentum=cfg.sdd.ema_momentum)

        # Projection heads (student / teacher 각각 독립)
        self.proj_student = ProjectionHead(
            in_dim=feat_dim,
            hidden_dim=cfg.sdd.proj_hidden_dim,
            out_dim=cfg.sdd.proj_out_dim,
        ).to(self.device)

        # teacher proj = EMA of student proj
        self.ema_proj_teacher = EMAModel(self.proj_student, momentum=cfg.sdd.ema_momentum)

        # ── 손실함수 ──────────────────────────────────────────────────────
        self.sdd_loss = SDDLoss(
            out_dim=cfg.sdd.proj_out_dim,
            tau_student=cfg.sdd.tau_student,
            tau_teacher=cfg.sdd.tau_teacher,
            center_momentum=cfg.sdd.center_momentum,
            centering_enabled=cfg.sdd.centering_enabled,
            gate_enabled=cfg.sdd.gate_enabled,
            gate_type=cfg.sdd.gate_type,
            t_min_ratio=cfg.sdd.t_min_ratio,
            t_max_ratio=cfg.sdd.t_max_ratio,
            gate_beta=cfg.sdd.gate_beta,
            lambda_dino=cfg.sdd.lambda_dino,
            T=cfg.diffusion.timesteps,
        ).to(self.device)

        # ── Diffusion schedule ─────────────────────────────────────────────
        self.schedule = DiffusionSchedule(
            timesteps=cfg.diffusion.timesteps,
            schedule=cfg.diffusion.beta_schedule,
            beta_start=cfg.diffusion.beta_start,
            beta_end=cfg.diffusion.beta_end,
        ).to(self.device)

        # ── Optimizer & Scheduler ─────────────────────────────────────────
        params = list(self.student.parameters()) + list(self.proj_student.parameters())
        self.optimizer = torch.optim.AdamW(
            params, lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
        )
        self.lr_sched = build_lr_scheduler(self.optimizer, cfg)

        # ── Mixed precision ───────────────────────────────────────────────
        self.scaler = GradScaler(enabled=cfg.training.mixed_precision)

        # ── 로거 ─────────────────────────────────────────────────────────
        self.logger = Logger(cfg, self.output_dir)

        # ── 체크포인트 재개 ───────────────────────────────────────────────
        self.step = 0
        ckpt_path = get_latest_checkpoint(self.output_dir)
        if ckpt_path:
            self.step = load_checkpoint(
                ckpt_path, self.student, self.ema_teacher,
                self.proj_student, self.ema_proj_teacher,
                self.optimizer, self.lr_sched, self.sdd_loss, self.device
            )
            print(f"Resumed from step {self.step}: {ckpt_path}")

    # ── 데이터 iter ──────────────────────────────────────────────────────

    def _next_batch(self):
        try:
            return next(self.train_iter)
        except StopIteration:
            self.train_iter = iter(self.train_loader)
            return next(self.train_iter)

    # ── 메인 학습 루프 ────────────────────────────────────────────────────

    def train(self):
        cfg = self.cfg
        total_steps = cfg.training.total_steps
        print(f"Training on {self.device} | total steps: {total_steps}")

        self.student.train()
        self.proj_student.train()

        while self.step < total_steps:
            # ── 배치 로드 ─────────────────────────────────────────────────
            v1, v2, _ = self._next_batch()
            # v1: student 입력 뷰, v2: teacher 입력 뷰 (서로 다른 증강)
            v1 = v1.to(self.device)
            v2 = v2.to(self.device)

            # ── 타임스텝 샘플링 ───────────────────────────────────────────
            t = torch.randint(0, cfg.diffusion.timesteps, (v1.shape[0],), device=self.device)

            # ── Forward process: x_t 생성 ──────────────────────────────────
            x_t_s, eps_s = self.schedule.q_sample(v1, t)   # student용 noisy image
            x_t_t, _     = self.schedule.q_sample(v2, t)   # teacher용 noisy image (다른 뷰)

            # ── Student forward ───────────────────────────────────────────
            with autocast(enabled=cfg.training.mixed_precision):
                eps_pred, z_s = self.student(x_t_s, t, return_features=True)
                z_s_proj = self.proj_student(z_s)             # [B, K]

                # ── Teacher forward (no gradient) ─────────────────────────
                with torch.no_grad():
                    _, z_t = self.ema_teacher(x_t_t, t, return_features=True)
                    z_t_proj = self.ema_proj_teacher(z_t)     # [B, K]

                # ── SDD Loss ──────────────────────────────────────────────
                loss_dict = self.sdd_loss(
                    noise_pred=eps_pred,
                    noise_target=eps_s,
                    student_proj=z_s_proj,
                    teacher_proj=z_t_proj,
                    t=t,
                )
                loss = loss_dict["total"]

            # ── Backward ─────────────────────────────────────────────────
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            if cfg.training.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    list(self.student.parameters()) + list(self.proj_student.parameters()),
                    cfg.training.grad_clip,
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.lr_sched.step()

            # ── EMA 갱신 (teacher & proj_teacher) ────────────────────────
            self.ema_teacher.update(self.student)
            self.ema_proj_teacher.update(self.proj_student)

            # ── 로깅 ─────────────────────────────────────────────────────
            self.logger.log(self.step, {
                "loss/total":     loss_dict["total"].item(),
                "loss/mse":       loss_dict["l_mse"].item(),
                "loss/dino":      loss_dict["l_dino"].item(),
                "gate/mean":      loss_dict["gate_mean"].item(),
                "lr":             self.lr_sched.get_last_lr()[0],
            })

            # ── 체크포인트 저장 ───────────────────────────────────────────
            if self.step > 0 and self.step % cfg.logging.save_every == 0:
                ckpt_path = os.path.join(
                    self.output_dir, f"ckpt_step_{self.step:07d}.pth"
                )
                save_checkpoint(
                    ckpt_path, self.step,
                    self.student, self.ema_teacher,
                    self.proj_student, self.ema_proj_teacher,
                    self.optimizer, self.lr_sched,
                    self.sdd_loss, cfg,
                )
                cleanup_old_checkpoints(self.output_dir, cfg.logging.keep_last_n)

            # ── 평가 ─────────────────────────────────────────────────────
            if self.step > 0 and self.step % cfg.evaluation.eval_every == 0:
                self._run_eval()

            self.step += 1

        print("Training complete.")

    def _run_eval(self):
        """FID + Linear Probe 평가 (evaluator.py에서 상세 구현)"""
        from trainer.evaluator import Evaluator
        evaluator = Evaluator(self.cfg, self.device)
        metrics = evaluator.evaluate(
            model=self.ema_teacher.shadow,
            proj_head=self.ema_proj_teacher.shadow,
            schedule=self.schedule,
            val_loader=self.val_loader,
            step=self.step,
        )
        self.logger.log_eval(self.step, metrics)
