"""
Multi-GPU training entry point via Hugging Face Accelerate.

Single GPU (or CPU):
    python scripts/train.py --config configs/cifar10.yaml

Multi-GPU (auto-detect all available GPUs, fp16):
    accelerate launch scripts/train.py --config configs/cifar10.yaml

Multi-GPU (explicit GPU count):
    accelerate launch --num_processes 2 scripts/train.py --config configs/cifar10.yaml

With eval after training:
    accelerate launch scripts/train.py --config configs/cifar10.yaml --eval
"""
from __future__ import annotations

import argparse
from pathlib import Path

from accelerate import Accelerator
from accelerate.utils import set_seed

from src.experiments.notebook_api import (
    load_cfg,
    build_loaders,
    build_trainer,
    make_optimizer,
    maybe_init_wandb,
    train_epochs,
    evaluate_generation,
    run_linear_probe,
    save_checkpoint,
)


def main():
    parser = argparse.ArgumentParser(description="Train SDD with Accelerate multi-GPU support.")
    parser.add_argument("--config",    type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--eval",      action="store_true",     help="Run FID + linear probe after training")
    parser.add_argument("--resume",    type=str, default=None,  help="Checkpoint path to resume from")
    parser.add_argument("--mixed_precision", type=str, default="fp16",
                        choices=["no", "fp16", "bf16"],
                        help="Mixed precision mode (default: fp16)")
    args = parser.parse_args()

    # ── Accelerator ──────────────────────────────────────────────────────────
    # num_processes is set automatically by `accelerate launch` based on
    # available GPUs. No config file needed.
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    is_main = accelerator.is_main_process

    if is_main:
        print(f"[accelerate] processes : {accelerator.num_processes}")
        print(f"[accelerate] device    : {accelerator.device}")
        print(f"[accelerate] mixed_prec: {accelerator.mixed_precision}")

    # ── Config + seed ─────────────────────────────────────────────────────────
    cfg = load_cfg(Path(args.config))
    set_seed(cfg["train"]["seed"])

    # ── Data ─────────────────────────────────────────────────────────────────
    train_loader, test_loader = build_loaders(cfg)

    # ── Model / optimizer ────────────────────────────────────────────────────
    trainer   = build_trainer(cfg, accelerator=accelerator)
    optimizer = make_optimizer(trainer, cfg)

    # Prepare: Accelerate wraps model in DDP, moves to device, sets up sampler
    optimizer, train_loader, test_loader = accelerator.prepare(
        optimizer, train_loader, test_loader
    )

    # ── Resume ───────────────────────────────────────────────────────────────
    if args.resume:
        from src.experiments.notebook_api import load_checkpoint
        if is_main:
            print(f"[train] resuming from {args.resume}")
        load_checkpoint(trainer, optimizer=optimizer, path=args.resume)

    # ── W&B (main process only) ───────────────────────────────────────────────
    run = maybe_init_wandb(cfg) if is_main else None

    # ── Training ─────────────────────────────────────────────────────────────
    history = train_epochs(
        trainer, train_loader, cfg, optimizer,
        run=run,
        val_loader=test_loader if args.eval else None,
        accelerator=accelerator,
    )

    # ── Post-training eval (main process only) ────────────────────────────────
    accelerator.wait_for_everyone()

    if args.eval and is_main:
        print("[train] evaluating FID ...")
        fid_result = evaluate_generation(trainer, test_loader, cfg, accelerator=accelerator)
        print(f"[train] FID = {fid_result.get('fid', 'N/A'):.2f}")

        print("[train] evaluating linear probe ...")
        acc = run_linear_probe(trainer, train_loader, test_loader, cfg, accelerator=accelerator)
        print(f"[train] linear probe acc = {acc:.4f}")

        if run is not None:
            run.log({"final_fid": fid_result.get("fid"), "final_probe_acc": acc})

    if run is not None:
        run.finish()

    accelerator.end_training()


if __name__ == "__main__":
    main()
