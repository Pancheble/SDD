# Self-Distilled Diffusion : Centering and Sharpening (DiDiCS)

**Self-Distilled Diffusion: Centering and Sharpening for Representation Learning in Generative Models**

---

## 프로젝트 구조

```
DiDiCS/
├── models/
│   ├── __init__.py
│   ├── unet.py                 # ADM-style UNet (feature 반환 지원)
│   ├── dit.py                  # DiT-B/4 (feature 반환 지원)
│   ├── projection_head.py      # 2-layer MLP projection head
│   └── ema.py                  # EMA 유틸리티
├── losses/
│   ├── __init__.py
│   ├── diffusion_loss.py       # MSE denoising loss
│   ├── dino_loss.py            # Centering + Sharpening cross-entropy
│   └── sdd_loss.py             # Combined SDD loss
├── data/
│   ├── __init__.py
│   ├── datasets.py             # CIFAR-10 / ImageNet 로더
│   └── augmentations.py        # 증강 파이프라인
├── trainer/
│   ├── __init__.py
│   ├── sdd_trainer.py          # 메인 학습 루프
│   └── evaluator.py            # FID / Linear Probe 평가
├── utils/
│   ├── __init__.py
│   ├── diffusion_utils.py      # noise schedule, sampling 유틸
│   ├── logging_utils.py        # WandB / TensorBoard 로거
│   └── checkpoint.py           # 체크포인트 저장/로드
├── scripts/
│   ├── train.py                # 학습 진입점
│   ├── evaluate.py             # FID + Linear Probe 평가
│   ├── ablation.py             # 절제 실험 일괄 실행
│   └── visualize_features.py  # CKA / entropy 시각화
├── experiments/
│   └── run_all.sh              # 전체 실험 재현 스크립트
└── requirements.txt
```