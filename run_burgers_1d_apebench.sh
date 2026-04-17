#!/bin/bash
#SBATCH --job-name=b1d_ape
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=05:50:00
#SBATCH --output=logs_1d3d/burgers_1d_apebench_%j.out
#SBATCH --error=logs_1d3d/burgers_1d_apebench_%j.err

cd /home/msai/song0304/code/PDE
python finetune/train_burgers_1d_apebench_lora.py --config configs/finetune_burgers_1d_apebench_rescaled_norm.yaml
