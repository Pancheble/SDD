"""
Linear probe evaluation with multi-GPU feature extraction.

Single GPU:
    python scripts/eval_linear.py --config configs/cifar10.yaml --checkpoint outputs/checkpoints/last.pt

Multi-GPU (features extracted in parallel across GPUs, gathered on main):
    accelerate launch scripts/eval_linear.py --config configs/cifar10.yaml --checkpoint outputs/checkpoints/last.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed

from src.experiments.notebook_api import (
    load_cfg,
    build_loaders,
    build_trainer_from_checkpoint,
)
from src.evaluation.linear_probe import train_linear_probe


def extract_features_distributed(model, loader, accelerator):
    """Extract features with all GPUs in parallel, gathered on all ranks."""
    model.eval()
    all_feats, all_labels = [], []

    with torch.no_grad():
        for x, y in loader:
            t = torch.zeros(x.size(0), device=accelerator.device, dtype=torch.long)
            _, f = model(x, t, return_features=True)

            # gather across GPUs
            f_all = accelerator.gather(f)
            y_all = accelerator.gather(y)

            all_feats.append(f_all.cpu())
            all_labels.append(y_all.cpu())

    return torch.cat(all_feats), torch.cat(all_labels)


def main():
    parser = argparse.ArgumentParser(description="Linear probe with Accelerate multi-GPU support.")
    parser.add_argument("--config",     type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--probe_epochs", type=int, default=50)
    parser.add_argument("--probe_lr",     type=float, default=1e-3)
    parser.add_argument("--mixed_precision", type=str, default="fp16",
                        choices=["no", "fp16", "bf16"])
    args = parser.parse_args()

    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    is_main     = accelerator.is_main_process

    cfg = load_cfg(Path(args.config))
    set_seed(cfg["train"]["seed"])

    trainer = build_trainer_from_checkpoint(cfg, args.checkpoint, accelerator=accelerator)
    raw_model = trainer._unwrap(trainer.model)

    train_loader, test_loader = build_loaders(cfg)
    train_loader, test_loader = accelerator.prepare(train_loader, test_loader)

    if is_main:
        print("[linear probe] extracting train features ...")
    train_feats, train_labels = extract_features_distributed(raw_model, train_loader, accelerator)

    if is_main:
        print("[linear probe] extracting test features ...")
    test_feats, test_labels = extract_features_distributed(raw_model, test_loader, accelerator)

    # Probe training only on main process (small enough to run single-threaded)
    accelerator.wait_for_everyone()
    if is_main:
        _, acc = train_linear_probe(
            train_feats=train_feats,
            train_labels=train_labels,
            val_feats=test_feats,
            val_labels=test_labels,
            num_classes=cfg["dataset"]["num_classes"],
            epochs=args.probe_epochs,
            lr=args.probe_lr,
            device=str(accelerator.device),
        )
        print(f"Linear probe accuracy = {acc:.4f}")


if __name__ == "__main__":
    main()
