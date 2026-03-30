# Self-Distilled Diffusion : Centering and Sharpening (DiDiCS)

**Self-Distilled Diffusion: Centering and Sharpening for Representation Learning in Generative Models**

---

## 프로젝트 구조

```
sdd_project/
├── configs/
│   ├── base.yaml               # 기본 설정
│   ├── cifar10.yaml            # CIFAR-10 실험
│   ├── imagenet.yaml           # ImageNet 실험
│   └── ablation/
│       ├── no_centering.yaml
│       ├── no_sharpening.yaml
│       ├── no_gate.yaml
│       └── ema_only.yaml
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
│   └── visualize_features.py   # CKA / entropy 시각화
├── experiments/
│   └── run_all.sh              # 전체 실험 재현 스크립트
└── requirements.txt
```

---

## 설치

```bash
pip install -r requirements.txt
```

---

## 빠른 시작

### CIFAR-10 학습

```bash
python scripts/train.py --config configs/cifar10.yaml
```

### ImageNet 학습

```bash
python scripts/train.py --config configs/imagenet.yaml --data_path /path/to/imagenet
```

### 평가 (FID + Linear Probe)

```bash
python scripts/evaluate.py --config configs/cifar10.yaml --ckpt outputs/cifar10/best.pth
```

### 절제 실험 전체 실행

```bash
bash experiments/run_all.sh
```

---

## 주요 하이퍼파라미터

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `tau_s` | 0.1 | 학생 온도 |
| `tau_t` | 0.04 | 교사 온도 (sharpening) |
| `ema_momentum` | 0.996 | EMA 모멘텀 |
| `center_momentum` | 0.9 | 센터링 EMA 모멘텀 |
| `lambda_dino` | 0.5 | DINO loss 가중치 |
| `t_min` | 0.1 | 타임스텝 게이트 하한 (비율) |
| `t_max` | 0.6 | 타임스텝 게이트 상한 (비율) |
| `proj_dim` | 256 | 투영 헤드 출력 차원 |

---