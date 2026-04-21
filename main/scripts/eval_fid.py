"""
FID evaluation with multi-GPU inference support.

Single GPU:
    python scripts/eval_fid.py --config configs/cifar10.yaml --checkpoint outputs/checkpoints/last.pt

Multi-GPU (each GPU generates a shard of fake samples, then FID is computed on main):
    accelerate launch scripts/eval_fid.py --config configs/cifar10.yaml --checkpoint outputs/checkpoints/last.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed

from src.evaluation.fid import FIDEvaluator
from src.experiments.notebook_api import (
    load_cfg,
    build_loaders,
    build_trainer_from_checkpoint,
)


def main():
    parser = argparse.ArgumentParser(description="Evaluate FID with Accelerate multi-GPU support.")
    parser.add_argument("--config",     type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Total fake samples to generate (default: cfg fid_num_samples)")
    parser.add_argument("--mixed_precision", type=str, default="fp16",
                        choices=["no", "fp16", "bf16"])
    args = parser.parse_args()

    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    is_main     = accelerator.is_main_process
    device      = accelerator.device

    cfg = load_cfg(Path(args.config))
    set_seed(cfg["train"]["seed"])

    # Load checkpoint (all ranks load weights; inference is independent per GPU)
    trainer = build_trainer_from_checkpoint(cfg, args.checkpoint, accelerator=accelerator)
    trainer._unwrap(trainer.model).eval()

    _, test_loader = build_loaders(cfg)
    test_loader    = accelerator.prepare(test_loader)

    total_samples = args.num_samples or cfg["train"].get("fid_num_samples", 2048)

    # ── Real images (streamed from test set) ─────────────────────────────────
    # Each process feeds its shard; we gather on main for FID
    fid = FIDEvaluator(device=device) if is_main else None

    seen = 0
    for x, _ in test_loader:
        # gather across GPUs so main gets everything
        x_all = accelerator.gather(x)
        if is_main:
            x_01 = (x_all * 0.5 + 0.5).clamp(0, 1)
            fid.update_real((x_01 * 255).to(torch.uint8))
            seen += x_01.size(0)
        if seen >= total_samples:
            break

    accelerator.wait_for_everyone()

    # ── Fake images: each GPU generates its share ─────────────────────────────
    samples_per_proc = (total_samples + accelerator.num_processes - 1) // accelerator.num_processes
    batch_size       = cfg["train"].get("num_samples_preview", 64)
    sample_shape     = (3, cfg["dataset"]["image_size"], cfg["dataset"]["image_size"])

    generated = []
    remaining = samples_per_proc
    while remaining > 0:
        n    = min(remaining, batch_size)
        fake = trainer.sample(n=n, shape=sample_shape)
        generated.append((fake * 0.5 + 0.5).clamp(0, 1))
        remaining -= n

    fake_local = torch.cat(generated, dim=0)   # (samples_per_proc, C, H, W)
    fake_all   = accelerator.gather(fake_local) # (total_samples, C, H, W) on all ranks

    if is_main:
        fid.update_fake((fake_all * 255).to(torch.uint8))
        score = fid.compute()
        print(f"FID = {score:.4f}  (over {total_samples} samples, {accelerator.num_processes} GPUs)")


if __name__ == "__main__":
    main()
