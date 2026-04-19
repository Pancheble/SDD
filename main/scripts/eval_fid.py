from __future__ import annotations

import argparse
from pathlib import Path

from src.experiments.notebook_api import build_loaders, build_trainer_from_checkpoint, evaluate_generation, load_cfg


def main():
    parser = argparse.ArgumentParser(description='Evaluate FID from a trained checkpoint.')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    cfg = load_cfg(Path(args.config))
    trainer = build_trainer_from_checkpoint(cfg, args.checkpoint, device=args.device)
    _, test_loader = build_loaders(cfg)
    print(evaluate_generation(trainer, test_loader, cfg))


if __name__ == '__main__':
    main()
