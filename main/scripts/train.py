"""
scripts/train.py
SDD 학습 진입점.

사용법:
    python scripts/train.py --config configs/cifar10.yaml
    python scripts/train.py --config configs/imagenet.yaml --data_path /path/to/imagenet
    python scripts/train.py --config configs/cifar10.yaml sdd.lambda_dino=1.0  # override
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import random
import numpy as np
import torch
from omegaconf import OmegaConf


def parse_args():
    parser = argparse.ArgumentParser(description="Train SDD")
    parser.add_argument("--config", type=str, required=True, help="Config YAML 경로")
    parser.add_argument("--data_path", type=str, default=None, help="데이터셋 경로 override")
    parser.add_argument("overrides", nargs="*", help="OmegaConf 스타일 override (key=value)")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()

    # ── Config 로드 ─────────────────────────────────────────────────────
    # base.yaml 먼저 로드 후 실험 config로 병합
    base_cfg = OmegaConf.load(
        os.path.join(os.path.dirname(args.config), "..", "configs", "base.yaml")
    )
    exp_cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(base_cfg, exp_cfg)

    # CLI override 적용
    if args.data_path:
        OmegaConf.update(cfg, "data.data_path", args.data_path)
    for override in args.overrides:
        key, val = override.split("=", 1)
        OmegaConf.update(cfg, key, val)

    print(OmegaConf.to_yaml(cfg))

    # ── 재현성 ──────────────────────────────────────────────────────────
    set_seed(cfg.seed)

    # ── 학습 ────────────────────────────────────────────────────────────
    from trainer.sdd_trainer import SDDTrainer
    trainer = SDDTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
