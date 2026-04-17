#!/bin/bash
#SBATCH --job-name=advection_3d
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=05:50:00
#SBATCH --output=logs_1d3d/advection_3d_%j.out
#SBATCH --error=logs_1d3d/advection_3d_%j.err

cd /home/msai/song0304/code/PDE
python finetune/train_advection_3d_lora_v3.py --config configs/finetune_advection_3d_v3_rescaled_norm.yaml
