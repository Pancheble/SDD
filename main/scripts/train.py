from __future__ import annotations

import argparse
from pathlib import Path

from src.experiments.notebook_api import load_cfg, run_experiment


def main():
    parser = argparse.ArgumentParser(description='Train an SDD experiment from a YAML config.')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--eval', action='store_true', help='Run FID and linear probe after training')
    args = parser.parse_args()

    cfg = load_cfg(Path(args.config))
    run_experiment(cfg, device=args.device, with_eval=args.eval)


if __name__ == '__main__':
    main()
