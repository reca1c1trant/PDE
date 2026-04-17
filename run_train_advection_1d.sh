#!/bin/bash
#SBATCH --job-name=train_adv1d
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=logs_train/advection_1d_%j.out
#SBATCH --error=logs_train/advection_1d_%j.err

cd /home/msai/song0304/code/PDE
mkdir -p logs_train

echo "=========================================="
echo "Train 1D Advection LoRA (V3)"
echo "Start: $(date)"
echo "Node: $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

source activate icml

torchrun --nproc_per_node=4 \
    finetune/train_advection_1d_lora_v3.py \
    --config configs/finetune_advection_1d_v3_rescaled_norm.yaml

echo "Done: $(date)"
