"""
scripts/ablation.py
논문 Table 2 (ablation study) 재현을 위한 일괄 실험 실행기.

실험 구성:
  1. baseline      : MSE only (sdd.enabled=False)
  2. ema_only      : EMA teacher, centering/sharpening 없음
  3. sharpening    : sharpening만 (centering 없음)
  4. centering     : centering만 (τ_t = τ_s)
  5. full_sdd      : 전체 SDD (centering + sharpening)
  6. no_gate       : 게이트 없는 SDD

사용법:
    python scripts/ablation.py --config configs/cifar10.yaml --gpu 0
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import subprocess
from omegaconf import OmegaConf


ABLATION_VARIANTS = {
    "baseline": [
        "sdd.enabled=false",
        "sdd.lambda_dino=0.0",
        "experiment_name=ablation_baseline",
    ],
    "ema_only": [
        "sdd.centering_enabled=false",
        "sdd.tau_teacher=0.1",   # τ_t = τ_s → sharpening 없음
        "experiment_name=ablation_ema_only",
    ],
    "sharpening_only": [
        "sdd.centering_enabled=false",
        "sdd.tau_teacher=0.04",
        "experiment_name=ablation_sharpening_only",
    ],
    "centering_only": [
        "sdd.centering_enabled=true",
        "sdd.tau_teacher=0.1",   # τ_t = τ_s → sharpening 없음
        "experiment_name=ablation_centering_only",
    ],
    "full_sdd": [
        "sdd.centering_enabled=true",
        "sdd.tau_teacher=0.04",
        "experiment_name=ablation_full_sdd",
    ],
    "no_gate": [
        "sdd.centering_enabled=true",
        "sdd.tau_teacher=0.04",
        "sdd.gate_enabled=false",
        "experiment_name=ablation_no_gate",
    ],
}


def run_variant(config: str, overrides: list, gpu: int):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = [
        sys.executable, "scripts/train.py",
        "--config", config,
    ] + overrides
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(overrides)}")
    print(f"{'='*60}")
    subprocess.run(cmd, env=env, check=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/cifar10.yaml")
    parser.add_argument("--gpu",    type=int, default=0)
    parser.add_argument("--variants", nargs="*", default=None,
                        help="실행할 변형 이름 (미지정 시 전체 실행)")
    return parser.parse_args()


def main():
    args = parse_args()
    variants = args.variants or list(ABLATION_VARIANTS.keys())

    results = {}
    for name in variants:
        if name not in ABLATION_VARIANTS:
            print(f"Unknown variant: {name}, skipping")
            continue
        overrides = ABLATION_VARIANTS[name]
        try:
            run_variant(args.config, overrides, args.gpu)
            results[name] = "done"
        except subprocess.CalledProcessError as e:
            results[name] = f"FAILED: {e}"

    print("\n=== Ablation Summary ===")
    for name, status in results.items():
        print(f"  {name:25s}: {status}")


if __name__ == "__main__":
    main()
