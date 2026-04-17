#!/bin/bash
#SBATCH --job-name=eval_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=00:10:00
#SBATCH --output=logs_1d3d/eval_test_%j.out
#SBATCH --error=logs_1d3d/eval_test_%j.err

cd /home/msai/song0304/code/PDE
python tools/eval_rollout.py \
    --config configs/finetune_burgers_2d_apebench_rescaled_norm.yaml \
    --checkpoint checkpoints_burgers_2d_apebench_rescaled_norm/best_lora.pt \
    --model_type lora
