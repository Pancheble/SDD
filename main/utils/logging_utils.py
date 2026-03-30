"""
utils/logging_utils.py
WandB / TensorBoard / 콘솔 로거.
"""
import os
import time
import logging
from collections import defaultdict
from typing import Dict, Any

logger = logging.getLogger(__name__)


class MetricTracker:
    """학습 중 지표 누적 및 평균 계산"""

    def __init__(self):
        self._sums   = defaultdict(float)
        self._counts = defaultdict(int)

    def update(self, metrics: Dict[str, float]):
        for k, v in metrics.items():
            self._sums[k]   += float(v)
            self._counts[k] += 1

    def mean(self) -> Dict[str, float]:
        return {k: self._sums[k] / self._counts[k] for k in self._sums}

    def reset(self):
        self._sums.clear()
        self._counts.clear()


class Logger:
    """통합 로거 (콘솔 + WandB 선택)"""

    def __init__(self, cfg, output_dir: str):
        self.use_wandb = cfg.logging.use_wandb
        self.log_every = cfg.logging.log_every
        self.output_dir = output_dir
        self.tracker = MetricTracker()
        self._start = time.time()

        # 파일 로거 설정
        os.makedirs(output_dir, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(os.path.join(output_dir, "train.log")),
            ],
        )

        if self.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=cfg.project_name,
                    name=cfg.experiment_name,
                    config=dict(cfg),
                )
                self.wandb = wandb
            except ImportError:
                logger.warning("wandb not installed — disabling wandb logging")
                self.use_wandb = False

    def log(self, step: int, metrics: Dict[str, Any]):
        self.tracker.update(metrics)
        if step % self.log_every == 0:
            avgs = self.tracker.mean()
            elapsed = time.time() - self._start
            msg = f"step {step:07d} | " + " | ".join(
                f"{k}: {v:.4f}" for k, v in sorted(avgs.items())
            ) + f" | elapsed: {elapsed:.0f}s"
            logger.info(msg)
            if self.use_wandb:
                self.wandb.log({**avgs, "step": step})
            self.tracker.reset()

    def log_eval(self, step: int, metrics: Dict[str, Any]):
        msg = f"[EVAL] step {step:07d} | " + " | ".join(
            f"{k}: {v:.4f}" for k, v in sorted(metrics.items())
        )
        logger.info(msg)
        if self.use_wandb:
            self.wandb.log({f"eval/{k}": v for k, v in metrics.items()} | {"step": step})
