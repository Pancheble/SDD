"""
scripts/evaluate.py
저장된 체크포인트로 FID + Linear Probe 평가.

사용법:
    python scripts/evaluate.py --config configs/cifar10.yaml --ckpt outputs/sdd_cifar10/ckpt_step_0200000.pth
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
from omegaconf import OmegaConf

from models import UNet, DiT, ProjectionHead, EMAModel
from utils import DiffusionSchedule
from data import build_dataloaders
from trainer.evaluator import Evaluator
from trainer.sdd_trainer import build_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt",   type=str, required=True, help="체크포인트 경로")
    parser.add_argument("--fid_samples", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    base_cfg = OmegaConf.load("configs/base.yaml")
    exp_cfg  = OmegaConf.load(args.config)
    cfg      = OmegaConf.merge(base_cfg, exp_cfg)

    if args.fid_samples:
        OmegaConf.update(cfg, "evaluation.fid_samples", args.fid_samples)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 모델 로드 ────────────────────────────────────────────────────────
    student, feat_dim = build_model(cfg, device)
    ema_teacher       = EMAModel(student, momentum=cfg.sdd.ema_momentum)
    proj_student      = ProjectionHead(
        in_dim=feat_dim,
        hidden_dim=cfg.sdd.proj_hidden_dim,
        out_dim=cfg.sdd.proj_out_dim,
    ).to(device)
    ema_proj = EMAModel(proj_student, momentum=cfg.sdd.ema_momentum)

    ckpt = torch.load(args.ckpt, map_location=device)
    ema_teacher.load_state_dict(ckpt["ema_state"])
    ema_proj.load_state_dict(ckpt["ema_proj_teacher_state"])
    step = ckpt["step"]
    print(f"Loaded checkpoint at step {step}")

    # ── 데이터 & 스케줄 ───────────────────────────────────────────────────
    _, val_loader = build_dataloaders(cfg)
    schedule = DiffusionSchedule(
        timesteps=cfg.diffusion.timesteps,
        schedule=cfg.diffusion.beta_schedule,
    ).to(device)

    # ── 평가 ─────────────────────────────────────────────────────────────
    evaluator = Evaluator(cfg, device)
    metrics = evaluator.evaluate(
        model=ema_teacher.shadow,
        proj_head=ema_proj.shadow,
        schedule=schedule,
        val_loader=val_loader,
        step=step,
    )

    print("\n=== Evaluation Results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
