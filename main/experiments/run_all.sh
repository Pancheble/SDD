#!/bin/bash
# experiments/run_all.sh
# 논문 Table 1 & 2 전체 실험 재현 스크립트
#
# 사용법:
#   bash experiments/run_all.sh
#   bash experiments/run_all.sh --gpu 0 --data_path /path/to/data

set -e

GPU=0
DATA_PATH="./data"

# 인자 파싱
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --gpu)        GPU="$2";       shift ;;
        --data_path)  DATA_PATH="$2"; shift ;;
    esac
    shift
done

export CUDA_VISIBLE_DEVICES=$GPU

echo "=================================================="
echo "  SDD 전체 실험 재현"
echo "  GPU: $GPU | Data: $DATA_PATH"
echo "=================================================="

# ── Table 1: 주요 결과 ─────────────────────────────────────────────────────

echo -e "\n[1/6] Baseline (MSE only)"
python scripts/train.py --config configs/cifar10.yaml \
    sdd.lambda_dino=0.0 \
    experiment_name=table1_baseline

echo -e "\n[2/6] DDAE (EMA only, no centering/sharpening)"
python scripts/train.py --config configs/cifar10.yaml \
    sdd.centering_enabled=false \
    sdd.tau_teacher=0.1 \
    experiment_name=table1_ddae

echo -e "\n[3/6] Full SDD (centering + sharpening)"
python scripts/train.py --config configs/cifar10.yaml \
    experiment_name=table1_sdd_full

# ── Table 2: Ablation ──────────────────────────────────────────────────────

echo -e "\n[4/6] Ablation: sharpening only"
python scripts/train.py --config configs/cifar10.yaml \
    sdd.centering_enabled=false \
    sdd.tau_teacher=0.04 \
    experiment_name=ablation_sharpening_only

echo -e "\n[5/6] Ablation: centering only"
python scripts/train.py --config configs/cifar10.yaml \
    sdd.centering_enabled=true \
    sdd.tau_teacher=0.1 \
    experiment_name=ablation_centering_only

echo -e "\n[6/6] Ablation: no timestep gate"
python scripts/train.py --config configs/cifar10.yaml \
    sdd.gate_enabled=false \
    experiment_name=ablation_no_gate

# ── 평가 ──────────────────────────────────────────────────────────────────

echo -e "\n=================================================="
echo "  평가 실행"
echo "=================================================="

for exp in table1_baseline table1_ddae table1_sdd_full \
           ablation_sharpening_only ablation_centering_only ablation_no_gate; do
    CKPT=$(ls outputs/${exp}/ckpt_step_*.pth 2>/dev/null | tail -1)
    if [ -z "$CKPT" ]; then
        echo "체크포인트 없음: $exp, 건너뜀"
        continue
    fi
    echo -e "\nEvaluating: $exp"
    python scripts/evaluate.py \
        --config configs/cifar10.yaml \
        --ckpt "$CKPT"
done

# ── 시각화 ────────────────────────────────────────────────────────────────

echo -e "\n시각화 생성..."
CKPT=$(ls outputs/table1_sdd_full/ckpt_step_*.pth 2>/dev/null | tail -1)
if [ -n "$CKPT" ]; then
    python scripts/visualize_features.py \
        --ckpt "$CKPT" \
        --config configs/cifar10.yaml \
        --output_dir figures/
fi

echo -e "\n=================================================="
echo "  모든 실험 완료."
echo "  결과: outputs/ | 시각화: figures/"
echo "=================================================="
