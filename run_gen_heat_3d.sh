#!/bin/bash
#SBATCH --job-name=gen_heat3d
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=02:00:00
#SBATCH --output=logs_gen/gen_heat_3d_%j.out
#SBATCH --error=logs_gen/gen_heat_3d_%j.err

cd /home/msai/song0304/code/PDE
mkdir -p logs_gen

echo "=========================================="
echo "Generate 3D Heat Dataset"
echo "Start: $(date)"
echo "Node: $(hostname)"
echo "=========================================="

source activate icml

echo "Config: N_GRID=64, N_T=21, dt=0.05, N_SAMPLES=50, nu in [0.005, 0.05]"
python tools/generate_heat_3d.py

echo "Done: $(date)"
