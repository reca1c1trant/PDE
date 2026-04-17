#!/bin/bash
#SBATCH --job-name=b2d_fresh
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=05:50:00
#SBATCH --output=logs_1d3d/burgers_2d_fresh_%j.out
#SBATCH --error=logs_1d3d/burgers_2d_fresh_%j.err

cd /home/msai/song0304/code/PDE
python finetune/train_burgers_2d_apebench_lora.py \
    --config configs/finetune_burgers_2d_apebench_rescaled_norm.yaml \
    --init_weights checkpoints_burgers_2d_ch_lora_v3_rescaled_norm/best_lora.pt
