#!/bin/bash
#SBATCH --job-name=eval_apebench
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=01:00:00
#SBATCH --output=logs_1d3d/eval_apebench_%j.out
#SBATCH --error=logs_1d3d/eval_apebench_%j.err

cd /home/msai/song0304/code/PDE

echo "========== 2D Burgers APEBench =========="
python tools/eval_rollout.py \
    --config configs/finetune_burgers_2d_apebench_rescaled_norm.yaml \
    --checkpoint checkpoints_burgers_2d_apebench_rescaled_norm/best_lora.pt \
    --model_type lora

echo ""
echo "========== 1D Burgers APEBench =========="
python tools/eval_rollout.py \
    --config configs/finetune_burgers_1d_apebench_rescaled_norm.yaml \
    --checkpoint checkpoints_burgers_1d_apebench_rescaled_norm/best_lora.pt \
    --model_type lora

echo ""
echo "========== 3D Burgers APEBench =========="
python tools/eval_rollout.py \
    --config configs/finetune_burgers_3d_apebench_rescaled_norm.yaml \
    --checkpoint checkpoints_burgers_3d_apebench_rescaled_norm/best_lora.pt \
    --model_type lora
