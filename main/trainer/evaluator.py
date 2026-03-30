"""
trainer/evaluator.py
FID-50K 측정 및 Linear Probe 정확도 평가.

논문 Section 4.1:
  - FID-50K (clean-fid 라이브러리 사용)
  - Linear Probe: EMA teacher의 projection 이전 특징으로 ImageNet 선형 분류기 학습
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np


class LinearProbe(nn.Module):
    def __init__(self, feat_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


class Evaluator:
    def __init__(self, cfg, device: torch.device):
        self.cfg    = cfg
        self.device = device

    @torch.no_grad()
    def _extract_features(self, model, loader):
        """EMA 모델에서 특징 벡터 추출"""
        model.eval()
        feats, labels = [], []
        for batch in tqdm(loader, desc="Extracting features", leave=False):
            if len(batch) == 3:
                x, _, y = batch
            else:
                x, y = batch
            x = x.to(self.device)
            # timestep = 0 (clean image)으로 특징 추출
            t = torch.zeros(x.shape[0], dtype=torch.long, device=self.device)
            _, feat = model(x, t, return_features=True)
            feats.append(feat.cpu())
            labels.append(y)
        return torch.cat(feats), torch.cat(labels)

    def _linear_probe(self, model, val_loader) -> float:
        """
        논문 Section 4.1: 100 epoch 선형 탐색.
        EMA teacher 특징 벡터 → 선형 분류기 → Top-1 정확도.
        """
        lp_cfg = self.cfg.evaluation.linear_probe
        if not lp_cfg.enabled:
            return 0.0

        print("Running linear probe evaluation...")
        # 특징 추출
        feats, labels = self._extract_features(model, val_loader)

        # L2 정규화
        feats = F.normalize(feats, dim=-1)

        num_classes = int(labels.max().item()) + 1
        feat_dim    = feats.shape[1]
        probe       = LinearProbe(feat_dim, num_classes).to(self.device)
        optimizer   = torch.optim.SGD(
            probe.parameters(),
            lr=lp_cfg.lr, momentum=0.9, weight_decay=0.0
        )

        dataset = TensorDataset(feats, labels)
        loader  = DataLoader(dataset, batch_size=lp_cfg.batch_size, shuffle=True)

        probe.train()
        for epoch in range(lp_cfg.epochs):
            for x_b, y_b in loader:
                x_b, y_b = x_b.to(self.device), y_b.to(self.device)
                loss = F.cross_entropy(probe(x_b), y_b)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # 정확도 측정
        probe.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x_b, y_b in loader:
                x_b, y_b = x_b.to(self.device), y_b.to(self.device)
                preds = probe(x_b).argmax(dim=-1)
                correct += (preds == y_b).sum().item()
                total   += y_b.size(0)

        return correct / total * 100.0

    def _compute_fid(self, model, schedule, n_samples: int) -> float:
        """
        FID 계산 (clean-fid 라이브러리).
        실제 학습 시 clean_fid 패키지가 필요합니다.
        """
        try:
            from cleanfid import fid
        except ImportError:
            print("clean-fid not installed, skipping FID. Install: pip install clean-fid")
            return -1.0

        print(f"Generating {n_samples} samples for FID...")
        model.eval()
        samples_dir = os.path.join(self.cfg.output_dir, self.cfg.experiment_name, "fid_samples")
        os.makedirs(samples_dir, exist_ok=True)

        from torchvision.utils import save_image
        batch_size = 64
        generated  = 0
        img_idx    = 0
        c          = self.cfg.data
        shape      = (batch_size, 3, c.image_size, c.image_size)

        while generated < n_samples:
            cur_bs = min(batch_size, n_samples - generated)
            cur_shape = (cur_bs, 3, c.image_size, c.image_size)
            samples = schedule.ddim_sample(
                model, cur_shape, self.device,
                ddim_steps=50, eta=0.0, progress=False
            )
            samples = (samples.clamp(-1, 1) + 1) / 2  # [0, 1]
            for img in samples:
                save_image(img, os.path.join(samples_dir, f"{img_idx:06d}.png"))
                img_idx += 1
            generated += cur_bs

        dataset_name = self.cfg.data.dataset.lower()
        fid_score = fid.compute_fid(
            samples_dir,
            dataset_name=dataset_name,
            dataset_res=c.image_size,
            dataset_split="train",
        )
        return fid_score

    def evaluate(self, model, proj_head, schedule, val_loader, step: int) -> dict:
        metrics = {}
        cfg_eval = self.cfg.evaluation

        # FID
        fid_score = self._compute_fid(model, schedule, cfg_eval.fid_samples)
        metrics["fid"] = fid_score

        # Linear Probe
        acc = self._linear_probe(model, val_loader)
        metrics["linear_probe_acc"] = acc

        print(f"[Step {step}] FID: {fid_score:.3f} | Linear Probe: {acc:.2f}%")
        return metrics
