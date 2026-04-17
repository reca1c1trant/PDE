#!/bin/bash
#SBATCH --job-name=pde_1d3d
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=24:00:00
#SBATCH --output=logs_1d3d/slurm_%j.out
#SBATCH --error=logs_1d3d/slurm_%j.err

cd /home/msai/song0304/code/PDE
mkdir -p logs_1d3d

echo "=========================================="
echo "1D/3D Training (1 GPU, sequential)"
echo "Start: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi -L | head -1)"
echo "=========================================="

# 1. 1D Burgers
echo "[1/4] 1D Burgers - $(date)"
python finetune/train_burgers_1d_lora_v3.py --config configs/finetune_burgers_1d_v3_rescaled_norm.yaml 2>&1 | tee logs_1d3d/burgers_1d.log
echo "[1/4] Done at $(date)"

# 2. 1D Advection
echo "[2/4] 1D Advection - $(date)"
python finetune/train_advection_1d_lora_v3.py --config configs/finetune_advection_1d_v3_rescaled_norm.yaml 2>&1 | tee logs_1d3d/advection_1d.log
echo "[2/4] Done at $(date)"

# 3. 1D Heat
echo "[3/4] 1D Heat - $(date)"
python finetune/train_heat_1d_lora_v3.py --config configs/finetune_heat_1d_v3_rescaled_norm.yaml 2>&1 | tee logs_1d3d/heat_1d.log
echo "[3/4] Done at $(date)"

# 4. 3D Advection
echo "[4/4] 3D Advection - $(date)"
python finetune/train_advection_3d_lora_v3.py --config configs/finetune_advection_3d_v3_rescaled_norm.yaml 2>&1 | tee logs_1d3d/advection_3d.log
echo "[4/4] Done at $(date)"

echo "=========================================="
echo "All done at $(date)"
echo "=========================================="
